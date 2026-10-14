"""Fool the AI — make a pretrained ImageNet model misclassify your photo.

No training. Uses torchvision pretrained ResNet-50 or MobileNetV3-Large.
Game loop:
  1. Snap / upload a photo -> model names it (baseline top-1).
  2. Add distortion via sliders (rotation, blur, noise, occlusion, colour jitter).
  3. If top-1 changes -> you fooled it. Lower distortion = higher score.
  4. "AI cheat" runs FGSM (gradient-sign attack, near-invisible noise).
  5. Grad-CAM shows where the model was looking before / after.
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import hashlib
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import models

st.set_page_config(page_title="Fool the AI", page_icon="🕵️", layout="wide")

# Preset gallery. Files in this directory are the offline dataset cache; URLs are
# used only to populate a missing file on a machine with an internet connection.
PRESET_CACHE_DIR = Path(__file__).parent / "assets" / "presets"
PRESETS: dict[str, tuple[str, str]] = {
    "🐶 Dog": ("dog.jpg", "https://github.com/pytorch/hub/raw/master/images/dog.jpg"),
    "🐱 Cat": ("cat.jpg", "https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg?width=640"),
    "🍌 Banana": ("banana.jpg", "https://commons.wikimedia.org/wiki/Special:FilePath/Banana-Single.jpg?width=640"),
    "☕ Coffee": ("coffee.jpg", "https://commons.wikimedia.org/wiki/Special:FilePath/A_small_cup_of_coffee.JPG?width=640"),
    "🚲 Bicycle": ("bicycle.jpg", "https://commons.wikimedia.org/wiki/Special:FilePath/Bicycle.jpg?width=640"),
    "🍓 Strawberry": ("strawberry.jpg", "https://commons.wikimedia.org/wiki/Special:FilePath/Strawberries.jpg?width=640"),
}

# ---------------------------------------------------------------- helpers ---

@st.cache_resource(show_spinner="Loading pretrained model (first run downloads weights)…")
def load_model(name: str):
    """Returns (model, weights, preprocess). No training — pretrained only."""
    if name.startswith("ResNet"):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        target_layer = model.layer4[-1]
    else:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        target_layer = model.features[-1]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights, weights.transforms(), target_layer


def predict_tensor(inp: torch.Tensor, model, labels):
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits[0], dim=0)
        top5_p, top5_i = torch.topk(probs, 5)
    results = [(labels[i.item()], float(p.item())) for p, i in zip(top5_p, top5_i)]
    return results, logits, inp


def predict(pil_img: Image.Image, model, preprocess, labels):
    return predict_tensor(preprocess(pil_img).unsqueeze(0), model, labels)


def image_fingerprint(pil_img: Image.Image) -> str:
    """Stable identity for clearing results that belong to a different photo."""
    rgb = pil_img.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.size}".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def apply_augmentations(pil_img: Image.Image, rot, blur, noise, occ, jitter, seed=0):
    """All distortions done on numpy arrays. Returns perturbed PIL image."""
    rng = np.random.default_rng(seed)
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    # 1. rotation (keep size, black border)
    if abs(rot) > 1e-6:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # 2. blur (odd kernel from 0..10 radius)
    if blur > 0:
        k = int(round(blur)) * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    # 3. gaussian noise
    if noise > 0:
        n = rng.normal(0, noise, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    # 4. occlusion — centred black square, side = occ * min(h,w)
    if occ > 0:
        side = int(occ * min(h, w))
        x0, y0 = (w - side) // 2, (h - side) // 2
        img[y0:y0 + side, x0:x0 + side] = 0
    # 5. colour jitter (brightness+contrast+saturation together)
    if jitter > 0:
        p = Image.fromarray(img)
        b = 1.0 + jitter * float(rng.uniform(-0.9, 0.9))
        c = 1.0 + jitter * float(rng.uniform(-0.9, 0.9))
        s = 1.0 + jitter * float(rng.uniform(-0.9, 0.9))
        p = ImageEnhance.Brightness(p).enhance(b)
        p = ImageEnhance.Contrast(p).enhance(c)
        p = ImageEnhance.Color(p).enhance(max(s, 0.0))
        img = np.array(p)
    return Image.fromarray(img)


def distortion_score(rot, blur, noise, occ, jitter) -> float:
    """Normalised 0..1 distortion. Weights sum to 1 — lower is better."""
    return (
        abs(rot) / 45.0 * 0.25
        + blur / 10.0 * 0.20
        + noise / 50.0 * 0.25
        + occ / 0.80 * 0.20
        + jitter / 1.0 * 0.10
    )


def game_score(dist: float, fooled: bool) -> int:
    if not fooled:
        return 0
    return max(0, int(round(1000 * (1.0 - dist))))


def fgsm_attack(pil_img, model, preprocess, target_label_idx: int, eps: float) -> tuple[Image.Image, torch.Tensor]:
    """Untargeted-style FGSM pushing AWAY from the baseline label.

    eps is in normalised-tensor units (0.01–0.05 looks near-invisible).
    """
    model.eval()
    x = preprocess(pil_img).unsqueeze(0)
    x.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, torch.tensor([target_label_idx]))
    model.zero_grad(set_to_none=True)
    loss.backward()
    pert = torch.sign(x.grad.data) * eps
    # Invert torchvision's documented weight normalization, clip in pixel space,
    # then re-normalize. `adv` remains the exact tensor optimized by FGSM.
    mean = torch.tensor(preprocess.mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(preprocess.std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x_pix = x.detach() * std + mean
    adv_pix = torch.clamp(x_pix + pert * std, 0, 1)
    adv = (adv_pix - mean) / std
    arr = (adv_pix.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    # The model transform is 224×224, but all UI comparisons should preserve
    # the uploaded/preset image geometry.
    return Image.fromarray(arr).resize(pil_img.size, Image.Resampling.BILINEAR), adv.detach()


class GradCAM:
    """Minimal Grad-CAM for one conv target layer.

    Uses a *temporary* forward hook (removed after each call) so Streamlit
    reruns don't stack duplicate hooks. Requires input grad enabled —
    handled internally, so callers must NOT wrap inputs in torch.no_grad().
    """

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def __call__(self, inp: torch.Tensor, class_idx: int) -> np.ndarray:
        feats: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []

        def _fwd(m, i, o):
            feats.append(o)
            # This hook only runs inside the gradient-enabled CAM pass below.
            # retain_grad avoids registering a tensor hook during ordinary
            # torch.no_grad() inference, where activations have no grad_fn.
            o.retain_grad()

        handle = self.layer.register_forward_hook(_fwd)
        try:
            with torch.set_grad_enabled(True):
                x = inp.detach().clone().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x)
                score = logits[0, class_idx]
                score.backward()
                grads.append(feats[0].grad.detach())
        finally:
            handle.remove()

        if not feats or not grads:
            raise RuntimeError(
                "Grad-CAM got no gradients. The input tensor must require grad "
                "(this is handled inside GradCAM — do not call it under torch.no_grad())."
            )
        w = grads[0].mean(dim=(2, 3), keepdim=True)
        cam = (w * feats[0].detach()).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def overlay_cam(pil_img: Image.Image, cam: np.ndarray) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    heat = cv2.resize(cam, (w, h))
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(((0.45 * heat + 0.55 * img)).astype(np.uint8))


@st.cache_data(show_spinner=False)
def load_preset_image(filename: str, url: str) -> Image.Image | None:
    """Load a bundled preset first; download and cache it only if missing."""
    path = PRESET_CACHE_DIR / filename
    try:
        if path.is_file():
            with Image.open(path) as image:
                return image.convert("RGB")

        # Keep the cache valid if a download is interrupted.
        request = urllib.request.Request(url, headers={"User-Agent": "fool-the-ai/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
        PRESET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        rgb.save(temporary_path, format="JPEG", quality=92)
        temporary_path.replace(path)
        return rgb
    except Exception:
        return None


# ------------------------------------------------------------------ UI ------

st.title("🕵️ Fool the AI")
st.caption("Pretrained ResNet / MobileNet (ImageNet, **no training**) — can you break it with the *least* distortion?")

with st.sidebar:
    st.header("⚙️ Setup")
    model_name = st.selectbox("Pretrained model", ["ResNet-50", "MobileNetV3-Large"])
    eps = st.slider("FGSM ε (cheat strength)", 0.005, 0.10, 0.02, 0.005,
                    help="Gradient-step size. ~0.02 is near-invisible but often flips the label.")
    show_cam = st.toggle("Show Grad-CAM", value=True)
    st.divider()
    st.markdown(
        "**Rules**\n"
        "- 📸 Webcam / upload / preset → AI names it (baseline).\n"
        "- 🎚️ Distort until the **top-1 label changes**.\n"
        "- 🏆 Score = `1000 × (1 − distortion)`. Less distortion = more points.\n"
        "- 🤖 *AI cheat* = FGSM gradient attack (tiny invisible noise)."
    )

model, weights, preprocess, target_layer = load_model(model_name)
# TorchVision ships the correct ImageNet labels with each pretrained weight set,
# so predictions retain names without a separate network request.
labels = list(weights.meta["categories"])
cam_explainer = GradCAM(model, target_layer)

# ---- Step 1: image in ----
st.subheader("1️⃣ Show the AI an object")
c1, c2 = st.columns(2)
with c1:
    shot = st.camera_input("📷 Webcam photo")
with c2:
    up = st.file_uploader("📤 …or upload", type=["jpg", "jpeg", "png", "webp"])

st.write("🎮 No camera? Pick a preset — instant play, no webcam needed:")
pcols = st.columns(len(PRESETS))
for i, name in enumerate(PRESETS):
    with pcols[i]:
        if st.button(name, key=f"preset_{i}", use_container_width=True,
                     type="primary" if st.session_state.get("preset", "🐶 Dog") == name else "secondary"):
            st.session_state["preset"] = name
            st.session_state.pop("adv", None)  # stale FGSM belongs to previous image
            st.session_state.pop("adv_top5", None)
preset_name = st.session_state.get("preset", "🐶 Dog")

src: Image.Image | None = None
src_kind = ""
if shot is not None:
    src = Image.open(shot).convert("RGB")
    src_kind = "webcam"
elif up is not None:
    src = Image.open(up).convert("RGB")
    src_kind = "upload"
else:
    with st.spinner(f"Loading preset {preset_name}…"):
        filename, url = PRESETS[preset_name]
        src = load_preset_image(filename, url)
    src_kind = f"preset {preset_name}"
    if src is None:
        st.warning("Could not download preset (offline?). Use webcam/upload instead.")
        st.stop()
st.caption(f"Playing with: **{src_kind}**")

# An attack is valid only for this exact image, model, and epsilon value.
attack_key = (image_fingerprint(src), model_name, float(eps))
if st.session_state.get("adv_attack_key") != attack_key:
    st.session_state.pop("adv", None)
    st.session_state.pop("adv_top5", None)
    st.session_state.pop("adv_attack_key", None)

# baseline prediction on the CLEAN image
base_top5, base_logits, base_inp = predict(src, model, preprocess, labels)
base_label, base_conf = base_top5[0]
base_idx = labels.index(base_label)

# ---- Step 2: sliders ----
st.subheader("2️⃣ Distort it — flip the label with as *little* change as possible")
s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    rot = st.slider("🔄 Rotation°", -45.0, 45.0, 0.0, 1.0)
with s2:
    blur = st.slider("🌫️ Blur", 0.0, 10.0, 0.0, 0.5)
with s3:
    noise = st.slider("📺 Noise σ", 0.0, 50.0, 0.0, 1.0)
with s4:
    occ = st.slider("⬛ Occlusion", 0.0, 0.80, 0.0, 0.05)
with s5:
    jitter = st.slider("🎨 Colour jitter", 0.0, 1.0, 0.0, 0.05)
seed = st.slider("🎲 Random seed (noise / colour)", 0, 99, 0, help="Same sliders, different random pattern.")

pert = apply_augmentations(src, rot, blur, noise, occ, jitter, seed)
pert_top5, _, pert_inp = predict(pert, model, preprocess, labels)
pert_label, pert_conf = pert_top5[0]

dist = distortion_score(rot, blur, noise, occ, jitter)
fooled = pert_label != base_label
score = game_score(dist, fooled)

# ---- Step 3: results ----
st.subheader("3️⃣ Did you fool it?")
r1, r2 = st.columns(2)
with r1:
    st.image(src, caption=f"Original → **{base_label}** ({base_conf:.1%})", use_container_width=True)
with r2:
    st.image(pert, caption=f"Distorted → **{pert_label}** ({pert_conf:.1%})", use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Baseline", f"{base_label}", f"{base_conf:.1%} conf")
m2.metric("After distortion", f"{pert_label}",
          delta=f"{pert_conf:.1%} conf · {'FOOLED ✅' if fooled else 'not fooled'}",
          delta_color="normal" if fooled else "off")
m3.metric("Distortion / Score", f"{dist:.3f}", f"🏆 {score} pts")

if fooled:
    st.balloons()
    st.success(f"✅ Fooled! `{base_label}` → `{pert_label}` with distortion {dist:.3f} → **{score} pts**. "
               "Try lowering the sliders to raise your score!")
else:
    st.warning("❌ Not fooled yet — the top-1 label is unchanged. Push a slider or try the AI cheat below.")

with st.expander("🔍 Top-5 predictions (before vs after)"):
    t1, t2 = st.columns(2)
    with t1:
        st.write("**Clean image**")
        st.bar_chart({l: c for l, c in base_top5})
    with t2:
        st.write("**Distorted image**")
        st.bar_chart({l: c for l, c in pert_top5})

# ---- Step 4: FGSM cheat ----
st.subheader("4️⃣ 🤖 AI cheat — FGSM gradient attack")
st.write("One gradient step on the pixels, pushing the image *away* from the baseline label. "
         "Often invisible to you, devastating to the model.")
if st.button("⚡ Run FGSM cheat", type="primary"):
    with st.spinner("Computing gradients…"):
        adv, adv_tensor = fgsm_attack(src, model, preprocess, base_idx, eps)
    # Do not preprocess the resized display image again: it is not the tensor
    # that FGSM optimized.
    adv_top5, _, _ = predict_tensor(adv_tensor, model, labels)
    adv_label, adv_conf = adv_top5[0]
    st.session_state["adv"] = adv
    st.session_state["adv_top5"] = adv_top5
    st.session_state["adv_attack_key"] = attack_key

if "adv" in st.session_state:
    adv = st.session_state["adv"]
    adv_top5 = st.session_state["adv_top5"]
    adv_label, adv_conf = adv_top5[0]
    a1, a2 = st.columns(2)
    with a1:
        st.image(src, caption=f"Clean → {base_label} ({base_conf:.1%})", use_container_width=True)
    with a2:
        st.image(adv, caption=f"FGSM (ε={eps}) → **{adv_label}** ({adv_conf:.1%})", use_container_width=True)
    if adv_label != base_label:
        st.success(f"🤖 Cheat worked: `{base_label}` → `{adv_label}` with ε={eps} (near-zero visible change).")
    else:
        st.info(f"Cheat failed at ε={eps} — raise ε in the sidebar and retry.")
    with st.expander("🔍 FGSM top-5"):
        st.bar_chart({l: c for l, c in adv_top5})
    # pixel-diff visual
    # Also tolerate an adversarial image cached by an earlier app version.
    adv_for_diff = adv.convert("RGB").resize(src.size, Image.Resampling.BILINEAR)
    diff = np.abs(np.array(src.convert("RGB")).astype(int) - np.array(adv_for_diff).astype(int))
    st.caption(f"Mean |pixel diff| after FGSM: {diff.mean():.2f} / 255 (lower = more invisible)")

# ---- Step 5: Grad-CAM ----
if show_cam:
    st.subheader("5️⃣ 🔥 Grad-CAM — where was the model looking?")
    target_img = pert if (abs(dist) > 1e-9) else src
    target_top5 = pert_top5 if (abs(dist) > 1e-9) else base_top5
    cam_target = st.selectbox("Explain which label?",
                              [l for l, _ in target_top5],
                              help="Heatmap is computed w.r.t. this class.")
    cam_idx = labels.index(cam_target)
    clean_inp = preprocess(src).unsqueeze(0)
    pert_in = preprocess(target_img).unsqueeze(0)
    with st.spinner("Computing Grad-CAM…"):
        cam_clean = cam_explainer(clean_inp, labels.index(base_label))
        cam_pert = cam_explainer(pert_in, cam_idx)
    g1, g2 = st.columns(2)
    with g1:
        st.image(overlay_cam(src, cam_clean),
                 caption=f"Looking for `{base_label}` (clean)", use_container_width=True)
    with g2:
        st.image(overlay_cam(target_img, cam_pert),
                 caption=f"Looking for `{cam_target}` (distorted)", use_container_width=True)
    st.caption("Red/yellow = regions that most support the label. A successful attack usually "
               "scatters or shifts this focus.")

st.divider()
st.caption("Models: torchvision pretrained weights (ResNet50-IMAGENET1K_V2 / MobileNetV3-Large-IMAGENET1K_V2). "
           "Scoring: distortion = 0.25·rot + 0.20·blur + 0.25·noise + 0.20·occ + 0.10·jitter (normalised); "
           "score = 1000·(1−distortion) when fooled, else 0.")

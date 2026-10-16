# 🕵️ Fool the AI

A webcam game where you try to break a **pretrained ImageNet classifier with the least distortion possible**.

Snap a photo (or pick a preset), and a pretrained **ResNet-50** or **MobileNetV3-Large** names it. Then distort the image with sliders — rotation, blur, noise, occlusion, colour jitter — until the **top-1 label flips**. Lower distortion = higher score.

Two XAI / adversarial-ML showpieces built in:

- 🤖 **AI cheat button** runs a real **FGSM** (Fast Gradient Sign Method) attack — near-invisible pixel noise that often flips the prediction.
- 🔥 **Grad-CAM heatmaps** show where the model was looking before vs. after your attack.

No training anywhere. Pure inference + gradients on frozen torchvision weights. First run downloads ~100 MB of weights; label names ship with the weights so everything works offline after that.

---

## 🎮 How the game works

```
1. Show the AI an object  →  model outputs baseline top-1 label (e.g. "Labrador retriever", 92%)
2. Distort with sliders   →  live re-prediction on the perturbed image
3. Top-1 changes?         →  ✅ FOOLED! Score = 1000 × (1 − distortion)
4. Want the easy way?     →  🤖 AI cheat = one FGSM gradient step away from baseline label
5. Curious why it broke?  →  🔥 Grad-CAM compares attention (clean vs. distorted)
```

**Win condition:** change only the top-1 label (top-5 can stay similar). A perfect score is 1000 (fooled with ~zero distortion — only FGSM gets close).

---

## ✨ Features

| Feature | Details |
|---|---|
| 📷 3 input modes | Webcam (`st.camera_input`), file upload (jpg/png/webp), 6 one-click presets |
| 🎚️ 5 distortions | Rotation, Gaussian blur, Gaussian noise, centre occlusion, colour jitter + random seed |
| 🏆 Distortion-aware scoring | Weighted 0–1 distortion; `score = round(1000 × (1 − dist))` when fooled, else 0 |
| 🤖 FGSM cheat | Untargeted gradient-sign attack, ε slider (0.005–0.10, default 0.02), mean `|pixel diff|` readout |
| 🔥 Grad-CAM | Per-class heatmaps (clean vs. distorted), selectable target label, JET overlay |
| 📊 Live feedback | Side-by-side images, top-1 metrics, balloons on success, top-5 bar charts |
| 📦 Offline-first | Presets bundled in `assets/presets/`; ImageNet labels embedded in torchvision weights |
| 🔄 Stale-state guards | FGSM result auto-invalidated on image / model / ε change via SHA-256 fingerprint |

---

## 🧠 Models

Switchable in the sidebar — both frozen (`eval()` + `requires_grad_(False)`), ImageNet-1K V2 weights:

| Model | Weights | Grad-CAM target layer | Notes |
|---|---|---|---|
| ResNet-50 (default) | `ResNet50_Weights.IMAGENET1K_V2` | `model.layer4[-1]` | More accurate, slower |
| MobileNetV3-Large | `MobileNet_V3_Large_Weights.IMAGENET1K_V2` | `model.features[-1]` | Faster, great for weak laptops |

Label names come from `weights.meta["categories"]` (1000 ImageNet classes) — no separate download.

---

## 🎚️ Distortions & scoring

All distortions are applied on NumPy/OpenCV arrays in this order: rotation → blur → noise → occlusion → colour jitter.

| Slider | Range | What it does |
|---|---|---|
| 🔄 Rotation° | −45 … +45 | `cv2.warpAffine`, black border, same size |
| 🌫️ Blur | 0 … 10 | Gaussian kernel `2·round(blur)+1` |
| 📺 Noise σ | 0 … 50 | Additive Gaussian, seeded RNG |
| ⬛ Occlusion | 0 … 0.80 | Centred black square, side = `occ × min(h, w)` |
| 🎨 Colour jitter | 0 … 1.0 | Random brightness/contrast/saturation ±90% × jitter |
| 🎲 Seed | 0 … 99 | Resamples the noise / jitter randomness |

**Distortion (0 = clean, ~1 = maxed):**

```
dist = |rot|/45 × 0.25 + blur/10 × 0.20 + noise/50 × 0.25 + occ/0.80 × 0.20 + jitter/1.0 × 0.10
```

**Score:** `0` if not fooled, else `max(0, round(1000 × (1 − dist)))`. Tip: small noise (σ 5–15) + tiny rotation (5–10°) fools more cheaply than heavy blur or big occlusion.

---

## 🤖 FGSM cheat — how it works

1. Preprocess the clean image to a 224×224 normalised tensor with `requires_grad=True`.
2. Cross-entropy loss against the **baseline** top-1 index, `loss.backward()`.
3. Perturbation = `sign(gradient) × ε`, applied in de-normalised pixel space, clipped to [0, 1], re-normalised.
4. The returned display image is resized back to the original geometry, but **scoring uses the exact optimised tensor** (`predict_tensor`), not a re-preprocess.

At ε ≈ 0.02 the mean `|pixel diff|` is typically 2–5/255 — invisible to you, decisive to the model. If the cheat fails, raise ε and retry.

## 🔥 Grad-CAM — how it works

Minimal single-layer Grad-CAM (`app.py:GradCAM`): temporary forward hook on the target conv layer → forward with grad enabled → `score.backward()` for the chosen class → channel-wise mean of gradients as weights → ReLU + normalise → JET overlay at 45% opacity.

- Left panel: clean image × baseline label attention.
- Right panel: distorted image × selectable label attention (defaults to distorted top-1).
- A successful attack usually **scatters or shifts** the red/yellow focus off the object.

Hooks are registered per-call and removed in `finally`, so Streamlit reruns never stack duplicates. Never call it under `torch.no_grad()` — it enables grad internally.

---

## 🚀 Run

**Requirements:** Python 3.10+ (3.11 recommended), webcam optional, ~500 MB free for torch + weights.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed `http://localhost:8501` URL. First run downloads ImageNet weights (~100 MB) once and caches them.

### Offline presets

Six playable images are bundled so the app works with no camera and no network:

`assets/presets/` → `dog.jpg` 🐶 · `cat.jpg` 🐱 · `banana.jpg` 🍌 · `coffee.jpg` ☕ · `bicycle.jpg` 🚲 · `strawberry.jpg` 🍓

If a preset file is deleted, the app downloads it once (20 s timeout, `fool-the-ai/1.0` user-agent) and atomically re-caches it (`*.tmp` → rename). If the download fails offline, it warns and falls back to webcam/upload. The `.gitignore` keeps these small JPGs tracked while ignoring large weights/datasets/media.

---

## 🗂️ Project structure

```
fool-the-ai/
├── app.py              # entire app: model loading, augmentations, FGSM, Grad-CAM, Streamlit UI
├── requirements.txt    # streamlit, torch, torchvision, opencv-python, Pillow, numpy
├── assets/presets/     # 6 bundled offline preset images (+ atomic .tmp cache writes)
├── memory/             # scratch dir (empty)
└── README.md
```

Key functions in `app.py`: `load_model` (cached pretrained weights) · `apply_augmentations` · `distortion_score` / `game_score` · `fgsm_attack` · `GradCAM` / `overlay_cam` · `load_preset_image` (cache-first + download fallback) · `image_fingerprint` (stale-attack guard).

---

## 🏅 Tips to get a high score

1. Start from a preset (Dog/Cat are the most confidently classified).
2. Nudge **Noise σ to ~8**, then add **Rotation ~8°** — check top-5 to see the runner-up closing in.
3. Prefer noise + rotation over blur/occlusion (cheaper per distortion point).
4. Use the **seed slider** — same sliders, different random draw, sometimes flips for free.
5. Try **MobileNetV3** — smaller model, easier to fool than ResNet-50.
6. Watch Grad-CAM: when heat moves off the object, you're one slider tick from fooled.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` / torch install fails | Use Python 3.11, `pip install --upgrade pip`, then `pip install -r requirements.txt` (CPU torch is default) |
| First run slow / downloads weights | Normal (~100 MB once); weights cache in torch hub dir afterwards |
| Webcam blank in browser | Grant camera permission; or use upload / presets instead |
| Preset won't load offline | File was deleted and there's no network — use webcam/upload, or restore `assets/presets/` from git |
| FGSM "failed at ε" | Raise ε in the sidebar (try 0.03–0.05) and re-run; some images need a bigger step |
| Grad-CAM error / no gradients | Don't wrap calls in `no_grad` — handled internally; reload the page if a stale hook survived |

---

## 📚 Concepts to explore

Adversarial examples (Szegedy 2013, Goodfellow FGSM 2014) · ImageNet transfer learning · Grad-CAM (Selvaraju 2017) · robustness vs. accuracy trade-off. Good classroom prompt: *why does invisible noise flip a 90%+ confident prediction, and what does the Grad-CAM shift tell you about the features the model relied on?*

PRs welcome — ideas: targeted FGSM, PGD, side-by-side model comparison, leaderboard, occlusion-position control.

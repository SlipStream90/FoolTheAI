# 🕵️ Fool the AI

Webcam game: a **pretrained** ImageNet model (ResNet-50 or MobileNetV3-Large, no training)
names your photo. You distort it with sliders until the top-1 label flips — with the
**least distortion possible** for the highest score.

- 🤖 **AI cheat** button runs an **FGSM** gradient-sign attack (near-invisible noise).
- 🔥 **Grad-CAM** heatmaps show where the model was looking before/after.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads ImageNet weights (~100 MB). ImageNet labels are included in
the selected TorchVision weights, so label names also work offline.

## Offline presets

The six playable image presets are bundled in `assets/presets/`, so they work
without a network connection. If a preset file is removed, the app attempts to
download it once and atomically caches it back into that directory.

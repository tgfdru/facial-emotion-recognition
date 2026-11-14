"""Gradio demo: upload a photo or use the webcam to get the predicted emotion,
class probabilities and a Grad-CAM overlay.

Usage:
    python app.py --checkpoint best_model.pth
"""
import argparse

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.gradcam import grad_cam
from src.model import CLASS_NAMES, IMG_SIZE, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
model = None  # set in main()


def predict_emotion(img: np.ndarray):
    if img is None:
        return "No image", {}, None

    pil = Image.fromarray(img).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred_label = CLASS_NAMES[int(np.argmax(probs))]

    # Grad-CAM on the last conv layer, overlaid on the original image
    cam, _ = grad_cam(x, model, model.conv3)
    cam_arr = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(pil.size))
    heatmap = (plt.cm.jet(cam_arr / 255.0)[..., :3] * 255).astype(np.uint8)
    overlay = (0.6 * np.array(pil, dtype=np.float32) + 0.4 * heatmap).astype(np.uint8)

    return pred_label, {c: float(p) for c, p in zip(CLASS_NAMES, probs)}, overlay


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict_emotion,
        inputs=gr.Image(sources=["webcam", "upload"], type="numpy", label="Image / Webcam"),
        outputs=[
            gr.Label(label="Predicted Emotion"),
            gr.Label(label="Probabilities"),
            gr.Image(label="Grad-CAM"),
        ],
        title="Human Face Emotion Recognition",
    )


def main() -> None:
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pth", help="Path to trained weights")
    args = parser.parse_args()
    model = load_model(args.checkpoint, device)
    build_demo().launch()


if __name__ == "__main__":
    main()

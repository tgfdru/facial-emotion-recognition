# 😃 Facial Emotion Recognition

**A from-scratch CNN in PyTorch that classifies facial expressions into five emotions, with Grad-CAM explainability and a live webcam demo.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Demo-F97316)
![Task](https://img.shields.io/badge/Task-Computer%20Vision-6A5ACD)

---

## Overview

| | |
|---|---|
| **Task** | Multi-class facial expression classification |
| **Classes** | Angry · Fear · Happy · Sad · Surprise |
| **Dataset** | [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) (Kaggle), ~59K images |
| **Split** | 80 / 20 train–validation (seeded) |
| **Input** | RGB, resized to 128×128 |

## Model: EmotionNet

A compact CNN trained from scratch, with no pretrained backbone:

```
Input (3×128×128)
 → Conv 3×3 (32)  + ReLU → MaxPool 2×2
 → Conv 3×3 (64)  + ReLU → MaxPool 2×2
 → Conv 3×3 (128) + ReLU → MaxPool 2×2
 → Flatten (128×14×14)
 → FC 128 + ReLU → Dropout 0.5
 → FC 5 (logits)
```

**Training:** Adam · CrossEntropyLoss · batch 32 · up to 30 epochs · early stopping (patience 5) · best-checkpoint saving on validation loss.

## Explainability: Grad-CAM

Grad-CAM on the last convolutional layer (`conv3`) shows which facial regions (eyes, mouth, brows) drove each prediction, so the model's decisions can be checked rather than taken on trust.

## Evaluation

The notebook reports validation accuracy, a per-class classification report (precision / recall / F1), a confusion matrix, training curves (accuracy and loss), and sample predictions with class-probability bars.

## Live demo (Gradio)

Upload a photo or use the **webcam** to get:
1. the predicted emotion
2. probabilities for all five classes
3. a Grad-CAM heatmap overlaid on the face

## Project structure

```
├── notebooks/
│   └── Face_Emotion_Recognition.ipynb   # Full pipeline: data → training → evaluation → Grad-CAM → demo
├── src/
│   ├── model.py      # EmotionNet + checkpoint loader
│   └── gradcam.py    # Grad-CAM implementation
├── app.py            # Gradio web/webcam demo
└── requirements.txt
```

## Quick start

1. **Train:** open the notebook in Google Colab (GPU runtime) and run all cells. The dataset downloads automatically with `kagglehub`, and the best weights are saved to `best_model.pth`.
2. **Run the demo** locally with the trained weights:

```bash
pip install -r requirements.txt
python app.py --checkpoint best_model.pth
```

## Tech stack

Python · PyTorch · torchvision · scikit-learn · Matplotlib · Seaborn · Gradio · kagglehub

## Author

**Nader Alshehri**, AI Engineer (ML, LLMs & AI Agents)
[LinkedIn](https://www.linkedin.com/in/nader-alshehri-15671b298)

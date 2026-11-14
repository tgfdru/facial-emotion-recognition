"""Grad-CAM: highlight the face regions that drove the model's prediction."""
import numpy as np
import torch
import torch.nn as nn


def grad_cam(img_tensor: torch.Tensor, model: nn.Module, target_layer: nn.Module):
    """Return (heatmap in [0, 1], predicted class index) for a single image tensor (1, 3, H, W)."""
    model.eval()
    activations, gradients = [], []

    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()
    h1.remove()
    h2.remove()

    acts = activations[0].detach().cpu().numpy()[0]
    grads = gradients[0].detach().cpu().numpy()[0]
    weights = grads.mean(axis=(1, 2))  # global-average-pooled gradients per channel

    cam = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0)
    cam = cam / (cam.max() + 1e-10)
    return cam.astype(np.float32), pred_class

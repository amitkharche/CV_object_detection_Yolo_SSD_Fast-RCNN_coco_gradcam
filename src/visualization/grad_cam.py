import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_index=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Handle different model output types
        if isinstance(output, tuple):
            # SSD: (class_preds, bbox_preds)
            class_preds = output[0]
            if target_index is None:
                target_index = torch.argmax(class_preds[0]).item()
            loss = class_preds[0, target_index]

        elif isinstance(output, list) and isinstance(output[0], dict):
            # Fast R-CNN: list of dicts with boxes, labels, scores
            scores = output[0].get("scores", None)
            if scores is None or len(scores) == 0:
                raise ValueError("Fast R-CNN returned no scores to use for Grad-CAM.")
            loss = scores[0]  # Use top scoring object

        elif isinstance(output, torch.Tensor):
            # YOLOv3 or other tensor-based output
            if target_index is None:
                target_index = output.argmax(dim=1).item()
            loss = output[0, target_index]

        else:
            raise TypeError(f"Unsupported model output type: {type(output)}")

        loss.backward()

        # Compute the Grad-CAM heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = (255 * cam).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap


def overlay_cam(image, heatmap, alpha=0.4):
    """Overlay the Grad-CAM heatmap on the original image."""
    image = np.array(image)

    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlay

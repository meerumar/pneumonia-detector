"""Grad-CAM: shows WHICH parts of the X-ray influenced the model's decision."""
import base64
import io

import numpy as np
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def __call__(self, x, class_idx=None):
        """Return (heatmap 0..1 as 2-D numpy array, raw logits)."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # importance of each feature map
        cam = F.relu((weights * self.activations).sum(dim=1))[0]       # weighted sum -> heatmap
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), logits.detach()


def overlay_heatmap(cam: np.ndarray, image: Image.Image, alpha: float = 0.45) -> str:
    """Blend the heatmap over the original image and return it as a base64 PNG data-URI."""
    image = image.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    blended = Image.fromarray((((1 - alpha) * base + alpha * heat) * 255).astype(np.uint8))
    blended.thumbnail((600, 600))
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

"""Grad-CAM explainability module.

Targets the last convolutional layer (backbone.conv_head) of EfficientNet-B0
as identified in Cell 10 of the research notebook, generating gradient-weighted
class activation heatmaps and blending them over the chest X-ray image.
"""

import time
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import settings
from .model_loader import model_manager
from .utils import (
    InferenceException,
    encode_cv2_to_base64_data_uri,
    encode_pil_to_base64_data_uri
)


class GradCAMGenerator:
    """Native PyTorch implementation of Grad-CAM targeting convolutional layers."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.target_layer = self._find_target_conv_layer()

    def _find_target_conv_layer(self) -> Tuple[nn.Module, str]:
        """Locate the target convolutional layer.

        Prefers 'backbone.conv_head' matching notebook Cell 10. Falls back
        to the final Conv2d layer if conv_head is not directly accessible.
        """
        # 1. Direct lookup matching notebook Cell 10: target_layers = [effnet.backbone.conv_head]
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "conv_head"):
            return self.model.backbone.conv_head, "backbone.conv_head"

        # 2. Dynamic traversal for the last 2D convolution layer
        last_conv = None
        last_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_name = name

        if last_conv is not None:
            return last_conv, last_name

        raise InferenceException(
            "Model architecture does not contain any convolutional layer suitable for Grad-CAM."
        )

    def _save_gradient(self, grad: torch.Tensor) -> None:
        """Gradient hook callback."""
        self.gradients = grad

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_index: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Compute the Grad-CAM heatmap array for the requested class.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
            target_class_index: Optional target class (0=NORMAL, 1=PNEUMONIA).
                                If None, defaults to the winning predicted class.

        Returns:
            Tuple of (2D float heatmap in [0, 1], target_class_index, target_class_probability).
        """
        layer_module, layer_name = self.target_layer

        # Hook to capture forward activation map
        def forward_hook(module, inp, out):
            self.activations = out
            # Register backward hook on the output tensor directly
            out.register_hook(self._save_gradient)

        hook_handle = layer_module.register_forward_hook(forward_hook)

        try:
            self.model.zero_grad()
            # Enable gradient for input tensor and forward pass
            tensor_with_grad = input_tensor.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                logits = self.model(tensor_with_grad)
                probs = F.softmax(logits, dim=1).squeeze(0)

                # Determine target class
                if target_class_index is None:
                    target_class_index = int(logits.argmax(dim=1).item())

                class_prob = float(probs[target_class_index].item())

                # Target score for gradient backpropagation
                score = logits[0, target_class_index]
                score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise InferenceException("Failed to capture activations or gradients during Grad-CAM backward pass.")

            # Global average pooling of gradients over spatial dimensions (H, W)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)  # Discard negative contributions

            cam_np = cam.squeeze().detach().cpu().numpy()

            # Normalize to [0.0, 1.0]
            cam_min = cam_np.min()
            cam_max = cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_normalized = np.zeros_like(cam_np)

            return cam_normalized, target_class_index, class_prob

        finally:
            hook_handle.remove()
            self.activations = None
            self.gradients = None
            self.model.zero_grad()


def generate_gradcam_explanation(
    input_tensor: torch.Tensor,
    original_pil_image: Image.Image,
    target_class_index: Optional[int] = None,
    overlay_alpha: float = 0.45
) -> Dict[str, Any]:
    """Generate Grad-CAM heatmap and blended overlay visualization.

    Args:
        input_tensor: Tensor of shape (1, 3, 224, 224).
        original_pil_image: PIL Image object (RGB).
        target_class_index: Target class index (0 for NORMAL, 1 for PNEUMONIA), or None.
        overlay_alpha: Blending weight for heatmap on top of X-ray (default 0.45).

    Returns:
        Structured explanation response matching ExplainResponse.

    Raises:
        InferenceException: If gradient generation or image encoding fails.
    """
    model = model_manager.get_model()
    start_time = time.perf_counter()

    try:
        # Acquire lock to safeguard model parameters during gradient backprop
        with model_manager.model_lock:
            model.eval()
            gradcam = GradCAMGenerator(model)
            heatmap_raw, class_idx, class_prob = gradcam.generate_heatmap(
                input_tensor,
                target_class_index=target_class_index
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Target class names
        class_name = settings.CLASS_INDEX_MAPPING.get(class_idx, f"CLASS_{class_idx}")

        # Convert original PIL image to RGB numpy array
        orig_w, orig_h = original_pil_image.size
        orig_np = np.array(original_pil_image)

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap_raw, (orig_w, orig_h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap (blue = low activation, red = high activation)
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend original chest X-ray with colored heatmap
        alpha = max(0.1, min(overlay_alpha, 0.9))
        overlay_rgb = cv2.addWeighted(
            orig_np,
            1.0 - alpha,
            heatmap_colored_rgb,
            alpha,
            0
        )

        # Convert RGB to BGR for OpenCV base64 encoder
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        heatmap_bgr = heatmap_colored_bgr

        overlay_base64 = encode_cv2_to_base64_data_uri(overlay_bgr, format="PNG")
        heatmap_base64 = encode_cv2_to_base64_data_uri(heatmap_bgr, format="PNG")

        return {
            "predicted_class": class_name,
            "target_class_explained": class_name,
            "target_class_index": class_idx,
            "target_layer": gradcam.target_layer[1],
            "overlay_base64": overlay_base64,
            "heatmap_base64": heatmap_base64,
            "inference_time_ms": round(elapsed_ms, 2),
            "message": (
                "Grad-CAM visualization highlights regions influencing model prediction. "
                "Research prototype output; does not prove clinical reasoning."
            ),
        }

    except Exception as exc:
        raise InferenceException(f"Error generating Grad-CAM visualization: {str(exc)}") from exc

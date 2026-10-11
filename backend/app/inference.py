"""Standard deterministic classification inference service."""

import time
from typing import Dict, Any

import torch
import torch.nn.functional as F

from .config import settings
from .model_loader import model_manager
from .utils import InferenceException


def run_classification_inference(input_tensor: torch.Tensor) -> Dict[str, Any]:
    """Execute a single deterministic forward pass on the input tensor.

    Uses torch.inference_mode() and model.eval() as required for safe,
    efficient inference without gradient tracking or buffer updates.

    Args:
        input_tensor: Tensor of shape (1, 3, 224, 224) on target device.

    Returns:
        Dictionary containing prediction results, probabilities, and timing.

    Raises:
        InferenceException: If an error occurs during tensor forward pass.
    """
    model = model_manager.get_model()
    device = model_manager.device

    start_time = time.perf_counter()

    try:
        with torch.inference_mode():
            # Thread-safe acquisition of model lock
            with model_manager.model_lock:
                model.eval()
                logits = model(input_tensor)

            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        normal_prob = float(probs[0])
        pneumo_prob = float(probs[1])

        predicted_idx = int(probs.argmax())
        predicted_class = settings.CLASS_INDEX_MAPPING[predicted_idx]
        confidence = float(probs[predicted_idx])

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": round(confidence, 4),
            "probabilities": {
                "NORMAL": round(normal_prob, 4),
                "PNEUMONIA": round(pneumo_prob, 4),
            },
            "model": settings.MODEL_NAME,
            "device": str(device),
            "inference_time_ms": round(elapsed_ms, 2),
        }

    except Exception as exc:
        raise InferenceException(f"Error during deterministic prediction: {str(exc)}") from exc

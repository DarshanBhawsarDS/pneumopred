"""Monte Carlo Dropout uncertainty quantification module.

Implements stochastic forward sampling with active dropout layers while
strictly preserving evaluation mode on BatchNorm layers. Computes predictive
entropy, aleatoric expected entropy, epistemic mutual information, and variance.
"""

import time
import math
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import settings
from .model_loader import model_manager
from .utils import InferenceException


def _activate_mc_dropout(model: nn.Module) -> None:
    """Set model to eval mode, then selectively activate Dropout layers only.

    Crucial for Monte Carlo Dropout:
    1. model.eval() disables running mean/variance computation in BatchNorm layers.
    2. Setting dropout layers to .train(True) enables stochastic sampling.
    """
    model.eval()  # Freeze all BatchNorm layers
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train(True)
        elif "dropout" in module.__class__.__name__.lower() or "droppath" in module.__class__.__name__.lower():
            module.train(True)


def run_mc_dropout_sampling(
    input_tensor: torch.Tensor,
    num_samples: int = settings.MC_SAMPLES_DEFAULT,
    user_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Execute Monte Carlo Dropout sampling and calculate uncertainty metrics.

    Args:
        input_tensor: Tensor of shape (1, 3, 224, 224) on target device.
        num_samples: Number of stochastic passes (T).
        user_threshold: Optional validation-derived uncertainty threshold for decision support.

    Returns:
        Structured uncertainty analysis results matching PredictWithUncertaintyResponse.

    Raises:
        InferenceException: If sampling fails during forward passes.
    """
    # Enforce safe bounds on stochastic sampling iterations
    num_samples = max(
        settings.MC_SAMPLES_MIN,
        min(num_samples, settings.MC_SAMPLES_MAX)
    )

    model = model_manager.get_model()
    device = model_manager.device
    start_time = time.perf_counter()

    try:
        # Acquire thread lock to avoid race conditions with standard inference
        with model_manager.model_lock:
            try:
                _activate_mc_dropout(model)

                stochastic_probs = []
                with torch.no_grad():
                    for _ in range(num_samples):
                        logits = model(input_tensor)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                        stochastic_probs.append(probs)

            finally:
                # Always restore model to clean evaluation mode
                model.eval()

        probs_array = np.array(stochastic_probs)  # Shape: (T, num_classes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 1. Mean Predictive Probabilities across T passes: \bar{p} = (1/T) sum p^(t)
        mean_probs = np.mean(probs_array, axis=0)
        mean_normal = float(mean_probs[0])
        mean_pneumo = float(mean_probs[1])

        predicted_idx = int(np.argmax(mean_probs))
        predicted_class = settings.CLASS_INDEX_MAPPING[predicted_idx]

        # Numerical stability epsilon
        eps = 1e-12

        # 2. Total Predictive Entropy: H(\bar{p}) = - sum \bar{p}_c log2(\bar{p}_c)
        # Unit: bits (log base 2). For 2 classes, maximum is 1.0 bit (when p = [0.5, 0.5]).
        predictive_entropy = float(-np.sum(mean_probs * np.log2(mean_probs + eps)))
        predictive_entropy = max(0.0, predictive_entropy)

        # 3. Expected Entropy (Aleatoric / Data Uncertainty): E[H(p)] = (1/T) sum H(p^(t))
        entropies_per_pass = -np.sum(probs_array * np.log2(probs_array + eps), axis=1)
        expected_entropy = float(np.mean(entropies_per_pass))
        expected_entropy = max(0.0, expected_entropy)

        # 4. Mutual Information (Epistemic / Model Uncertainty): I = H(\bar{p}) - E[H(p)]
        # Measures how much information the model parameters provide about the label.
        mutual_info = float(max(0.0, predictive_entropy - expected_entropy))

        # 5. Predictive Variance & Standard Deviation across passes
        variances = np.var(probs_array, axis=0)
        stds = np.std(probs_array, axis=0)

        pred_variance = {
            "NORMAL": round(float(variances[0]), 6),
            "PNEUMONIA": round(float(variances[1]), 6),
        }
        mean_variance = round(float(np.mean(variances)), 6)

        pred_std = {
            "NORMAL": round(float(stds[0]), 6),
            "PNEUMONIA": round(float(stds[1]), 6),
        }

        # 6. Clinical Decision Support / Review Recommendation
        # The research notebook did not calibrate a clinically validated threshold.
        # We allow a user-supplied or configured threshold, else mark experimental.
        effective_threshold = user_threshold if user_threshold is not None else settings.UNCERTAINTY_THRESHOLD

        if effective_threshold is not None:
            threshold_configured = True
            threshold_val = float(effective_threshold)
            if predictive_entropy >= threshold_val:
                recommendation = "FLAGGED_FOR_HUMAN_REVIEW"
                note = (
                    f"Predictive entropy ({predictive_entropy:.4f} bits) exceeds configured research "
                    f"threshold ({threshold_val:.4f} bits). Second opinion or radiologist review recommended."
                )
            else:
                recommendation = "CONFIDENT_PREDICTION"
                note = (
                    f"Predictive entropy ({predictive_entropy:.4f} bits) is below configured research "
                    f"threshold ({threshold_val:.4f} bits). Research prototype output; not a clinical guarantee."
                )
        else:
            threshold_configured = False
            threshold_val = None
            recommendation = "EXPERIMENTAL_UNREVIEWED"
            note = (
                "No validation-derived clinical threshold was established in the notebook. "
                "Predictive uncertainty metrics are provided for experimental research analysis only. "
                "High entropy/mutual information indicates elevated model uncertainty."
            )

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "mean_probabilities": {
                "NORMAL": round(mean_normal, 4),
                "PNEUMONIA": round(mean_pneumo, 4),
            },
            "uncertainty": {
                "predictive_entropy": round(predictive_entropy, 4),
                "expected_entropy": round(expected_entropy, 4),
                "mutual_information": round(mutual_info, 4),
                "predictive_variance": pred_variance,
                "mean_variance": mean_variance,
                "standard_deviation": pred_std,
                "entropy_unit": "bits (log2)",
                "num_stochastic_passes": num_samples,
            },
            "decision_support": {
                "review_recommendation": recommendation,
                "threshold_configured": threshold_configured,
                "threshold_value": threshold_val,
                "note": note,
            },
            "model": settings.MODEL_NAME,
            "device": str(device),
            "inference_time_ms": round(elapsed_ms, 2),
        }

    except Exception as exc:
        raise InferenceException(f"Error during Monte Carlo Dropout sampling: {str(exc)}") from exc

"""Model architecture and thread-safe lifecycle manager.

Implements the exact EfficientNetB0Classifier defined in the research notebook
and manages one-time checkpoint loading at startup.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

from .config import settings

logger = logging.getLogger("pneumopred.model_loader")


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 transfer learning classifier with custom dropout head.

    Recreated exactly from Cell 7 of 'ChestXRay_Pneumonia_Detection_v2.ipynb'.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.35, pretrained: bool = False):
        super().__init__()
        if timm is None:
            raise ImportError(
                "The 'timm' library is required to build EfficientNetB0Classifier. "
                "Install it via: pip install timm"
            )

        # Backbone: timm efficientnet_b0 with pooled feature extractor
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 1280 for EfficientNet-B0

        # Classification Head matching trained checkpoint weights
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier head."""
        features = self.backbone(x)
        return self.classifier(features)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities for input tensor."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class ModelManager:
    """Singleton model lifecycle manager.

    Loads weights once during application startup, preserves thread-safety
    during stochastic MC Dropout passes and Grad-CAM backward operations.
    """

    def __init__(self):
        self.model: Optional[EfficientNetB0Classifier] = None
        self.device: torch.device = torch.device("cpu")
        self.checkpoint_path: Optional[Path] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None
        self.model_lock = threading.Lock()
        self.metadata: Dict[str, Any] = {}

    def get_device(self) -> torch.device:
        """Resolve computation device based on settings and CUDA availability."""
        if settings.DEVICE:
            device_str = settings.DEVICE.lower()
            if device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested via DEVICE setting but not available. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(device_str)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> bool:
        """Load the model architecture and restore checkpoint weights.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        self.device = self.get_device()
        logger.info(f"Using compute device: {self.device}")

        ckpt_path = settings.resolve_checkpoint_path()
        if ckpt_path is None or not ckpt_path.exists():
            self.is_loaded = False
            self.load_error = (
                f"Checkpoint file not found. Checked default locations and MODEL_CHECKPOINT_PATH. "
                f"Ensure 'efficientnet_b0_best.pth' is placed in 'backend/models/' or specified via .env."
            )
            logger.warning(self.load_error)
            return False

        self.checkpoint_path = ckpt_path
        logger.info(f"Found checkpoint at: {ckpt_path}")

        try:
            # Instantiate architecture without re-downloading ImageNet weights
            model = EfficientNetB0Classifier(
                num_classes=settings.NUM_CLASSES,
                dropout=settings.EFFNET_DROPOUT,
                pretrained=False
            )

            # Load checkpoint dictionary safely
            checkpoint_data = torch.load(
                str(ckpt_path),
                map_location=self.device,
                weights_only=False
            )

            # Handle both raw state_dict and container dict with metadata
            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
                self.metadata = {
                    k: v for k, v in checkpoint_data.items() if k != "model_state_dict"
                }
            elif isinstance(checkpoint_data, dict):
                state_dict = checkpoint_data
            else:
                raise ValueError(f"Unrecognized checkpoint format of type: {type(checkpoint_data)}")

            # Clean module prefix if saved under DataParallel / DistributedDataParallel
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v

            model.load_state_dict(cleaned_state_dict, strict=True)
            model.to(self.device)
            model.eval()  # Freeze BatchNorm and Dropout for default inference

            self.model = model
            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded model checkpoint and initialized in evaluation mode.")
            return True

        except Exception as exc:
            self.is_loaded = False
            self.load_error = f"Failed to load checkpoint from {ckpt_path}: {str(exc)}"
            logger.error(self.load_error, exc_info=True)
            return False

    def get_model(self) -> EfficientNetB0Classifier:
        """Get the loaded model instance.

        Raises:
            ModelNotLoadedException: If the model has not been loaded.
        """
        if not self.is_loaded or self.model is None:
            from .utils import ModelNotLoadedException
            raise ModelNotLoadedException(
                self.load_error or "Model is not loaded. Please verify model checkpoint configuration."
            )
        return self.model


# Global singleton instance
model_manager = ModelManager()

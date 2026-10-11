"""Application configuration module.

Reads settings from environment variables with sensible defaults matching
the research notebook 'ChestXRay_Pneumonia_Detection_v2.ipynb'.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings


# Base paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Configuration settings for the Pneumonia Uncertainty Detection backend."""

    # Project metadata
    PROJECT_NAME: str = "Uncertainty-Aware Pneumonia Detection API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = False

    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Model and Architecture configuration (from notebook Cell 5 & Cell 7)
    MODEL_NAME: str = "EfficientNet-B0 (timm)"
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["NORMAL", "PNEUMONIA"]
    CLASS_INDEX_MAPPING: dict = {0: "NORMAL", 1: "PNEUMONIA"}
    IMG_SIZE: int = 224
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
    EFFNET_DROPOUT: float = 0.35
    CONV_TARGET_LAYER: str = "backbone.conv_head"

    # Hardware & Inference
    DEVICE: Optional[str] = None  # None for auto-detection ('cuda' if available else 'cpu')

    # Model Checkpoint Path
    MODEL_CHECKPOINT_PATH: Optional[str] = None

    # Monte Carlo Dropout Uncertainty settings
    MC_SAMPLES_DEFAULT: int = 20
    MC_SAMPLES_MIN: int = 2
    MC_SAMPLES_MAX: int = 100
    # Validation-derived threshold: Notebook did not establish a clinical threshold.
    # We keep it configurable, default None (marked experimental/unconfigured).
    UNCERTAINTY_THRESHOLD: Optional[float] = None

    # Security & File Upload limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    ALLOWED_IMAGE_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def resolve_checkpoint_path(self) -> Optional[Path]:
        """Resolve the path to the trained model checkpoint.

        Searches in:
        1. Explicit MODEL_CHECKPOINT_PATH environment variable
        2. backend/models/efficientnet_b0_best.pth
        3. backend/models/efficientnet_b0_pneumonia.pth
        4. Workspace root: d:/pneumopred/efficientnet_b0_best.pth
        """
        candidates: List[Path] = []

        if self.MODEL_CHECKPOINT_PATH:
            candidates.append(Path(self.MODEL_CHECKPOINT_PATH))

        candidates.extend([
            BACKEND_DIR / "models" / "efficientnet_b0_best.pth",
            BACKEND_DIR / "models" / "efficientnet_b0_pneumonia.pth",
            WORKSPACE_DIR / "efficientnet_b0_best.pth",
            Path("d:/pneumopred/efficientnet_b0_best.pth"),
        ])

        for path in candidates:
            if path.exists() and path.is_file():
                return path.resolve()

        return None


settings = Settings()

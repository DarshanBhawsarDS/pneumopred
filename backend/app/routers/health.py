"""Health check and model information API endpoints."""

from fastapi import APIRouter
from ..config import settings
from ..model_loader import model_manager
from ..schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["System & Model Metadata"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return the health status of the API service and loaded model.

    Crucial guarantee: Does NOT report model as loaded if checkpoint loading failed.
    """
    is_loaded = model_manager.is_loaded
    ckpt_path = str(model_manager.checkpoint_path) if model_manager.checkpoint_path else None

    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_architecture=settings.MODEL_NAME,
        device=str(model_manager.device),
        checkpoint_path=ckpt_path,
        version=settings.VERSION,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Return model specification, input dimensions, class mapping, and verified capabilities."""
    ckpt_exists = (
        model_manager.checkpoint_path is not None and
        model_manager.checkpoint_path.exists()
    )

    ckpt_size_bytes = (
        model_manager.checkpoint_path.stat().st_size if ckpt_exists else 0
    )

    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        architecture=(
            "EfficientNet-B0 backbone with custom classification head: "
            "BatchNorm1d(1280) -> Dropout(0.35) -> Linear(1280, 512) -> "
            "SiLU -> Dropout(0.175) -> Linear(512, 2)"
        ),
        supported_classes=settings.CLASS_NAMES,
        class_to_idx={name: idx for idx, name in settings.CLASS_INDEX_MAPPING.items()},
        input_dimensions={
            "width": settings.IMG_SIZE,
            "height": settings.IMG_SIZE,
            "channels": 3,
            "color_space": "RGB",
        },
        normalization={
            "mean": settings.IMAGENET_MEAN,
            "std": settings.IMAGENET_STD,
        },
        checkpoint_status={
            "is_loaded": model_manager.is_loaded,
            "path": str(model_manager.checkpoint_path) if model_manager.checkpoint_path else None,
            "file_exists": ckpt_exists,
            "file_size_mb": round(ckpt_size_bytes / (1024 * 1024), 2) if ckpt_exists else 0.0,
            "load_error": model_manager.load_error,
        },
        uncertainty_estimation_available=True,  # Verified Monte Carlo Dropout implementation
        gradcam_explainability_available=True,  # Verified Grad-CAM targeting backbone.conv_head
        device=str(model_manager.device),
    )

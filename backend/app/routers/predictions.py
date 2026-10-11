"""Prediction, Uncertainty, and Explainability API endpoints."""

import time
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse

from ..config import settings
from ..model_loader import model_manager
from ..preprocessing import validate_and_read_image, preprocess_image
from ..inference import run_classification_inference
from ..uncertainty import run_mc_dropout_sampling
from ..explainability import generate_gradcam_explanation
from ..schemas import (
    PredictionResponse,
    PredictWithUncertaintyResponse,
    ExplainResponse,
    AnalyzeResponse,
    ErrorResponse,
)
from ..utils import (
    ModelNotLoadedException,
    InvalidImageException,
    InferenceException,
)

router = APIRouter(tags=["Inference & Diagnostics"])


def _check_model_readiness():
    """Ensure model checkpoint is loaded before handling inference requests."""
    if not model_manager.is_loaded or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. Please ensure 'efficientnet_b0_best.pth' is present "
                "in 'backend/models/' and check /api/health for details."
            )
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG or PNG)")
) -> PredictionResponse:
    """Classify a chest X-ray image as NORMAL or PNEUMONIA using the trained EfficientNet-B0 model."""
    _check_model_readiness()

    try:
        file_bytes = await file.read()
        pil_image = validate_and_read_image(file_bytes, filename=file.filename)
        input_tensor, _ = preprocess_image(pil_image, model_manager.device)

        result = run_classification_inference(input_tensor)
        return PredictionResponse(**result)

    except InvalidImageException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InferenceException as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/predict-with-uncertainty",
    response_model=PredictWithUncertaintyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict_with_uncertainty(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG or PNG)"),
    num_samples: int = Query(
        settings.MC_SAMPLES_DEFAULT,
        ge=settings.MC_SAMPLES_MIN,
        le=settings.MC_SAMPLES_MAX,
        description="Number of Monte Carlo Dropout stochastic passes"
    ),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optional experimental uncertainty entropy threshold in bits [0.0, 1.0]"
    )
) -> PredictWithUncertaintyResponse:
    """Predict pneumonia with Monte Carlo Dropout uncertainty estimation.

    Performs stochastic sampling with active dropout layers to compute
    predictive entropy, aleatoric expected entropy, and epistemic mutual information.
    """
    _check_model_readiness()

    try:
        file_bytes = await file.read()
        pil_image = validate_and_read_image(file_bytes, filename=file.filename)
        input_tensor, _ = preprocess_image(pil_image, model_manager.device)

        result = run_mc_dropout_sampling(
            input_tensor,
            num_samples=num_samples,
            user_threshold=threshold
        )
        return PredictWithUncertaintyResponse(**result)

    except InvalidImageException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InferenceException as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def explain_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG or PNG)"),
    target_class: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Target class index to explain (0=NORMAL, 1=PNEUMONIA). Defaults to winning prediction."
    ),
    alpha: float = Query(
        0.45,
        ge=0.1,
        le=0.9,
        description="Heatmap transparency overlay weight"
    )
) -> ExplainResponse:
    """Generate a Grad-CAM saliency heatmap and chest X-ray overlay.

    Hooks into the final convolutional head ('backbone.conv_head') to visualize
    spatial image regions that influenced the model's diagnostic score.
    """
    _check_model_readiness()

    try:
        file_bytes = await file.read()
        pil_image = validate_and_read_image(file_bytes, filename=file.filename)
        input_tensor, resized_pil = preprocess_image(pil_image, model_manager.device)

        result = generate_gradcam_explanation(
            input_tensor=input_tensor,
            original_pil_image=resized_pil,
            target_class_index=target_class,
            overlay_alpha=alpha
        )
        return ExplainResponse(**result)

    except InvalidImageException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InferenceException as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def analyze_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (JPEG or PNG)"),
    include_uncertainty: bool = Query(
        True,
        description="Whether to perform Monte Carlo Dropout uncertainty analysis"
    ),
    include_explainability: bool = Query(
        True,
        description="Whether to generate Grad-CAM visualization"
    ),
    num_samples: int = Query(
        settings.MC_SAMPLES_DEFAULT,
        ge=settings.MC_SAMPLES_MIN,
        le=settings.MC_SAMPLES_MAX,
        description="Number of MC Dropout stochastic passes if uncertainty is requested"
    ),
    uncertainty_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optional experimental uncertainty entropy threshold in bits"
    ),
    target_class: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Target class index for Grad-CAM if explainability is requested"
    )
) -> AnalyzeResponse:
    """Execute combined diagnostic pipeline: classification, uncertainty metrics, and Grad-CAM.

    Convenient single-request endpoint for frontend applications.
    """
    _check_model_readiness()

    total_start = time.perf_counter()

    try:
        file_bytes = await file.read()
        pil_image = validate_and_read_image(file_bytes, filename=file.filename)
        input_tensor, resized_pil = preprocess_image(pil_image, model_manager.device)

        # 1. Deterministic Classification
        base_result = run_classification_inference(input_tensor)

        uncertainty_metrics = None
        decision_support = None
        explanation_data = None

        # 2. Optional Uncertainty Quantification
        if include_uncertainty:
            unc_result = run_mc_dropout_sampling(
                input_tensor=input_tensor,
                num_samples=num_samples,
                user_threshold=uncertainty_threshold
            )
            uncertainty_metrics = unc_result["uncertainty"]
            decision_support = unc_result["decision_support"]

        # 3. Optional Grad-CAM Explainability
        if include_explainability:
            # If target class not explicitly requested, default to predicted class
            target_idx = target_class if target_class is not None else base_result["predicted_index"]
            exp_result = generate_gradcam_explanation(
                input_tensor=input_tensor,
                original_pil_image=resized_pil,
                target_class_index=target_idx,
                overlay_alpha=0.45
            )
            explanation_data = {
                "target_class_explained": exp_result["target_class_explained"],
                "target_layer": exp_result["target_layer"],
                "overlay_base64": exp_result["overlay_base64"],
                "heatmap_base64": exp_result["heatmap_base64"],
            }

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        return AnalyzeResponse(
            success=True,
            predicted_class=base_result["predicted_class"],
            predicted_index=base_result["predicted_index"],
            confidence=base_result["confidence"],
            probabilities=base_result["probabilities"],
            uncertainty=uncertainty_metrics,
            decision_support=decision_support,
            explanation=explanation_data,
            model=settings.MODEL_NAME,
            device=str(model_manager.device),
            inference_time_ms=round(total_elapsed_ms, 2),
            message="Research prototype output; not a clinical diagnosis."
        )

    except InvalidImageException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except InferenceException as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

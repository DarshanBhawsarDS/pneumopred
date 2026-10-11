"""Pydantic schemas for API request and response data contracts."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    model_loaded: bool = Field(..., description="Whether the model weights are successfully loaded")
    model_architecture: str = Field(..., description="Model architecture identifier")
    device: str = Field(..., description="Compute device being used ('cuda' or 'cpu')")
    checkpoint_path: Optional[str] = Field(None, description="Path to active checkpoint file if loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    model_name: str = Field(..., description="Canonical model name")
    architecture: str = Field(..., description="Neural network architecture details")
    supported_classes: List[str] = Field(..., description="List of target class names in index order")
    class_to_idx: Dict[str, int] = Field(..., description="Mapping from class name to integer index")
    input_dimensions: Dict[str, Any] = Field(..., description="Required input image dimensions")
    normalization: Dict[str, List[float]] = Field(..., description="ImageNet normalization mean and std")
    checkpoint_status: Dict[str, Any] = Field(..., description="Checkpoint loading and file status")
    uncertainty_estimation_available: bool = Field(..., description="True if Monte Carlo Dropout is supported")
    gradcam_explainability_available: bool = Field(..., description="True if Grad-CAM explainability is supported")
    device: str = Field(..., description="Current computation device")


class PredictionResponse(BaseModel):
    """Single standard classification prediction response."""
    success: bool = Field(True, description="Whether prediction succeeded")
    predicted_class: str = Field(..., description="Predicted class name ('NORMAL' or 'PNEUMONIA')")
    predicted_index: int = Field(..., description="Predicted class index (0 or 1)")
    confidence: float = Field(..., description="Probability of the winning class")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device used for computation")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    message: str = Field(
        "Research prototype output; not a clinical diagnosis.",
        description="Medical research disclaimer"
    )


class UncertaintyMetrics(BaseModel):
    """Monte Carlo Dropout uncertainty metrics."""
    predictive_entropy: float = Field(
        ...,
        description="Total predictive uncertainty H(p) = -sum(p * log2(p)). Range [0, 1] bit for 2 classes."
    )
    expected_entropy: float = Field(
        ...,
        description="Aleatoric / data uncertainty E[H(p)] averaged over stochastic passes."
    )
    mutual_information: float = Field(
        ...,
        description="Epistemic / model uncertainty I = H(p) - E[H(p)]. High MI signals model ignorance."
    )
    predictive_variance: Dict[str, float] = Field(
        ...,
        description="Variance across stochastic passes per class."
    )
    mean_variance: float = Field(
        ...,
        description="Average predictive variance across all classes."
    )
    standard_deviation: Dict[str, float] = Field(
        ...,
        description="Standard deviation across stochastic passes per class."
    )
    entropy_unit: str = Field(
        "bits (log2)",
        description="Entropy measurement unit (log2 corresponds to bits; maximum 1.0 for binary)."
    )
    num_stochastic_passes: int = Field(
        ...,
        description="Number of Monte Carlo forward passes sampled."
    )


class DecisionSupport(BaseModel):
    """Uncertainty-guided decision support details."""
    review_recommendation: str = Field(
        ...,
        description="Clinical workflow recommendation ('EXPERIMENTAL_UNREVIEWED', 'FLAGGED_FOR_HUMAN_REVIEW', 'CONFIDENT_PREDICTION')"
    )
    threshold_configured: bool = Field(
        ...,
        description="Whether a validation-derived uncertainty threshold is active"
    )
    threshold_value: Optional[float] = Field(
        None,
        description="Numerical threshold value applied if configured"
    )
    note: str = Field(
        ...,
        description="Clarification distinguishing predictive uncertainty from calibrated clinical diagnosis"
    )


class PredictWithUncertaintyResponse(BaseModel):
    """Monte Carlo Dropout uncertainty prediction response."""
    success: bool = Field(True, description="Whether analysis succeeded")
    predicted_class: str = Field(..., description="Predicted class from mean probabilities")
    predicted_index: int = Field(..., description="Predicted class index (0 or 1)")
    mean_probabilities: Dict[str, float] = Field(
        ...,
        description="Stochastic mean probability distribution across passes"
    )
    uncertainty: UncertaintyMetrics = Field(..., description="Computed uncertainty metrics")
    decision_support: DecisionSupport = Field(..., description="Decision support recommendation")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device used for computation")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    message: str = Field(
        "Research prototype output; not a clinical diagnosis.",
        description="Medical research disclaimer"
    )


class ExplainResponse(BaseModel):
    """Grad-CAM explanation response."""
    success: bool = Field(True, description="Whether explanation succeeded")
    predicted_class: str = Field(..., description="Class predicted by model")
    target_class_explained: str = Field(..., description="Class name explained by the Grad-CAM heatmap")
    target_class_index: int = Field(..., description="Class index explained")
    target_layer: str = Field(..., description="Convolutional layer hooked for Grad-CAM")
    overlay_base64: str = Field(
        ...,
        description="Base64 Data URI of original image blended with Grad-CAM heatmap ('data:image/png;base64,...')"
    )
    heatmap_base64: Optional[str] = Field(
        None,
        description="Base64 Data URI of standalone colorized Grad-CAM heatmap"
    )
    inference_time_ms: float = Field(..., description="Explainability computation latency in milliseconds")
    message: str = Field(
        "Grad-CAM visualization highlights regions influencing model prediction. Research prototype output; does not prove clinical reasoning.",
        description="Medical research disclaimer"
    )


class AnalyzeResponse(BaseModel):
    """Combined analysis response combining classification, uncertainty, and Grad-CAM."""
    success: bool = Field(True, description="Whether combined analysis succeeded")
    predicted_class: str = Field(..., description="Predicted class")
    predicted_index: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    uncertainty: Optional[UncertaintyMetrics] = Field(
        None,
        description="Uncertainty metrics if requested"
    )
    decision_support: Optional[DecisionSupport] = Field(
        None,
        description="Decision support recommendation if uncertainty was computed"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="Grad-CAM visualization outputs if requested"
    )
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device used for computation")
    inference_time_ms: float = Field(..., description="Total execution latency in milliseconds")
    message: str = Field(
        "Research prototype output; not a clinical diagnosis.",
        description="Medical research disclaimer"
    )


class ErrorResponse(BaseModel):
    """Structured error response schema."""
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="High-level error classification")
    detail: str = Field(..., description="Detailed description of the issue")

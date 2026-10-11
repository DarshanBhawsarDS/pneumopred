"""FastAPI main application entrypoint."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from .config import settings
from .model_loader import model_manager
from .routers import health, predictions
from .utils import (
    ModelNotLoadedException,
    InvalidImageException,
    InferenceException,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pneumopred.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: loads model weights once at startup."""
    logger.info("Initializing Uncertainty-Aware Pneumonia Detection Backend...")
    loaded = model_manager.load_model()
    if loaded:
        logger.info(
            f"Model loaded successfully on {model_manager.device} "
            f"from {model_manager.checkpoint_path}"
        )
    else:
        logger.warning(
            f"Model could not be loaded at startup: {model_manager.load_error}. "
            "Inference endpoints will return 503 until a valid checkpoint is placed."
        )

    yield

    logger.info("Shutting down Pneumonia Detection Backend service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "**Uncertainty-Aware Pneumonia Detection API**\n\n"
        "College Research Prototype Backend using Transfer Learning on EfficientNet-B0.\n\n"
        "- **Classification:** Predicts NORMAL vs PNEUMONIA with calibrated probabilities.\n"
        "- **Uncertainty Estimation:** Monte Carlo Dropout sampling computes predictive entropy, "
        "aleatoric expected entropy, and epistemic mutual information.\n"
        "- **Explainability:** Grad-CAM saliency heatmaps visualizing convolutional activation on chest X-rays.\n"
        "- **Safety:** Clear medical disclaimers, no fabricated clinical conclusions."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(InvalidImageException)
async def invalid_image_handler(request: Request, exc: InvalidImageException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": "InvalidImageError", "detail": str(exc)},
    )


@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"success": False, "error": "ModelNotLoadedError", "detail": str(exc)},
    )


@app.exception_handler(InferenceException)
async def inference_error_handler(request: Request, exc: InferenceException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "InferenceError", "detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "ValidationError",
            "detail": exc.errors(),
        },
    )


# Mount routers under /api
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(predictions.router, prefix=settings.API_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")

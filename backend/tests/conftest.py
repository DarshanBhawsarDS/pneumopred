"""Pytest configuration, mock fixtures, and test data generators."""

import io
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import model_manager


class LightweightMockBackbone(nn.Module):
    """Mock backbone mimicking timm's EfficientNet-B0 with conv_head."""
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv_head = nn.Conv2d(16, 32, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.conv_head(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class LightweightMockClassifier(nn.Module):
    """Mock classifier mimicking EfficientNetB0Classifier for fast, CPU-only unit testing."""
    def __init__(self):
        super().__init__()
        self.backbone = LightweightMockBackbone()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Dropout(0.35),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(0.175),
            nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Instantiate a lightweight CPU-compatible mock model."""
    mock = LightweightMockClassifier()
    mock.eval()
    return mock


@pytest.fixture
def set_mock_loaded_model(mock_model):
    """Temporarily inject lightweight mock model into model_manager for tests."""
    original_model = model_manager.model
    original_loaded = model_manager.is_loaded
    original_path = model_manager.checkpoint_path
    original_device = model_manager.device
    original_error = model_manager.load_error

    model_manager.model = mock_model
    model_manager.is_loaded = True
    model_manager.checkpoint_path = Path("mock/efficientnet_b0_best.pth")
    model_manager.device = torch.device("cpu")
    model_manager.load_error = None

    yield mock_model

    # Restore state
    model_manager.model = original_model
    model_manager.is_loaded = original_loaded
    model_manager.checkpoint_path = original_path
    model_manager.device = original_device
    model_manager.load_error = original_error


@pytest.fixture
def set_mock_unloaded_model():
    """Temporarily simulate an unloaded model / missing checkpoint state."""
    original_model = model_manager.model
    original_loaded = model_manager.is_loaded
    original_error = model_manager.load_error

    model_manager.model = None
    model_manager.is_loaded = False
    model_manager.load_error = "Checkpoint not found (Simulated for test)."

    yield

    # Restore state
    model_manager.model = original_model
    model_manager.is_loaded = original_loaded
    model_manager.load_error = original_error


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate in-memory valid JPEG chest X-ray mock image."""
    img = Image.new("RGB", (224, 224), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    # Draw simple thoracic cavity shapes
    draw.ellipse([40, 40, 100, 180], fill=(80, 80, 80), outline=(120, 120, 120))
    draw.ellipse([124, 40, 184, 180], fill=(80, 80, 80), outline=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate in-memory valid PNG image."""
    img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Generate corrupted non-image binary bytes."""
    return b"\x89PNG\r\n\x1a\ncorrupted-random-binary-content-truncated\x00\x01\x02"


@pytest.fixture
def text_file_bytes() -> bytes:
    """Generate plain text content (invalid file format)."""
    return b"This is a text log file, not an image."

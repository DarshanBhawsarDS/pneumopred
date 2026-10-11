"""Tests for Health and Model Information endpoints."""

from fastapi import status


def test_health_loaded_model(client, set_mock_loaded_model):
    """Test /api/health reports healthy and loaded when model is active."""
    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert "EfficientNet" in data["model_architecture"]
    assert data["checkpoint_path"] is not None


def test_health_unloaded_model(client, set_mock_unloaded_model):
    """Test /api/health truthfully reports model_loaded=False when checkpoint is missing."""
    response = client.get("/api/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_model_info_endpoint(client, set_mock_loaded_model):
    """Test /api/model-info returns verified metadata and input dimensions."""
    response = client.get("/api/model-info")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["supported_classes"] == ["NORMAL", "PNEUMONIA"]
    assert data["class_to_idx"] == {"NORMAL": 0, "PNEUMONIA": 1}
    assert data["input_dimensions"]["width"] == 224
    assert data["input_dimensions"]["height"] == 224
    assert data["input_dimensions"]["channels"] == 3
    assert data["uncertainty_estimation_available"] is True
    assert data["gradcam_explainability_available"] is True
    assert "mean" in data["normalization"]
    assert "std" in data["normalization"]


def test_root_redirect_to_docs(client):
    """Test root URL redirects to interactive Swagger documentation."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [status.HTTP_307_TEMPORARY_REDIRECT, status.HTTP_302_FOUND]
    assert response.headers["location"] == "/docs"

"""Tests for Prediction, Uncertainty, Explainability, and Combined Analyze endpoints."""

import pytest
from fastapi import status


def test_predict_valid_jpeg(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test standard prediction endpoint with valid JPEG chest X-ray image."""
    response = client.post(
        "/api/predict",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in ["NORMAL", "PNEUMONIA"]
    assert data["predicted_index"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0

    probs = data["probabilities"]
    assert "NORMAL" in probs
    assert "PNEUMONIA" in probs
    # Probabilities must sum close to 1.0
    prob_sum = probs["NORMAL"] + probs["PNEUMONIA"]
    assert pytest.approx(prob_sum, abs=1e-2) == 1.0

    assert "inference_time_ms" in data
    assert "Research prototype output" in data["message"]


def test_predict_valid_png(client, set_mock_loaded_model, sample_png_bytes):
    """Test standard prediction endpoint with valid PNG image."""
    response = client.post(
        "/api/predict",
        files={"file": ("test_xray.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["success"] is True


def test_predict_missing_model_checkpoint(client, set_mock_unloaded_model, sample_jpeg_bytes):
    """Test /api/predict returns 503 Service Unavailable when model is not loaded."""
    response = client.post(
        "/api/predict",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Model is not loaded" in response.json()["detail"]


def test_predict_invalid_file_extension(client, set_mock_loaded_model, text_file_bytes):
    """Test /api/predict rejects invalid file extension with 400 Bad Request."""
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", text_file_bytes, "text/plain")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported file extension" in response.json()["detail"]


def test_predict_corrupted_image(client, set_mock_loaded_model, corrupt_image_bytes):
    """Test /api/predict safely handles corrupted image binary with 400 Bad Request."""
    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", corrupt_image_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Could not decode image" in response.json()["detail"]


def test_predict_with_uncertainty(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test Monte Carlo Dropout uncertainty quantification endpoint."""
    response = client.post(
        "/api/predict-with-uncertainty?num_samples=10",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in ["NORMAL", "PNEUMONIA"]

    unc = data["uncertainty"]
    assert "predictive_entropy" in unc
    assert "expected_entropy" in unc
    assert "mutual_information" in unc
    assert "predictive_variance" in unc
    assert "mean_variance" in unc
    assert "standard_deviation" in unc
    assert unc["num_stochastic_passes"] == 10
    assert unc["entropy_unit"] == "bits (log2)"

    # Epistemic MI = Predictive Entropy - Expected Entropy >= 0
    assert unc["predictive_entropy"] >= 0.0
    assert unc["expected_entropy"] >= 0.0
    assert unc["mutual_information"] >= 0.0

    # Decision support without threshold
    decision = data["decision_support"]
    assert decision["threshold_configured"] is False
    assert decision["review_recommendation"] == "EXPERIMENTAL_UNREVIEWED"


def test_predict_with_uncertainty_threshold(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test uncertainty endpoint with a user-configured threshold."""
    response = client.post(
        "/api/predict-with-uncertainty?num_samples=5&threshold=0.3",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    decision = data["decision_support"]
    assert decision["threshold_configured"] is True
    assert decision["threshold_value"] == 0.3
    assert decision["review_recommendation"] in ["FLAGGED_FOR_HUMAN_REVIEW", "CONFIDENT_PREDICTION"]


def test_explain_gradcam(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test Grad-CAM explainability endpoint returns Base64 overlay and heatmap."""
    response = client.post(
        "/api/explain?alpha=0.5",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in ["NORMAL", "PNEUMONIA"]
    assert data["target_class_explained"] in ["NORMAL", "PNEUMONIA"]
    assert data["overlay_base64"].startswith("data:image/png;base64,")
    assert data["heatmap_base64"].startswith("data:image/png;base64,")
    assert "conv_head" in data["target_layer"]


def test_analyze_combined_endpoint(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test combined analysis endpoint returning classification, uncertainty, and Grad-CAM."""
    response = client.post(
        "/api/analyze?include_uncertainty=true&include_explainability=true&num_samples=5",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "probabilities" in data
    assert data["uncertainty"] is not None
    assert data["decision_support"] is not None
    assert data["explanation"] is not None
    assert data["explanation"]["overlay_base64"].startswith("data:image/png;base64,")


def test_analyze_without_optional_features(client, set_mock_loaded_model, sample_jpeg_bytes):
    """Test combined analysis endpoint with optional features disabled."""
    response = client.post(
        "/api/analyze?include_uncertainty=false&include_explainability=false",
        files={"file": ("test_xray.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["success"] is True
    assert data["uncertainty"] is None
    assert data["decision_support"] is None
    assert data["explanation"] is None

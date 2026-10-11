# Uncertainty-Aware Pneumonia Detection Backend

Backend API service for the research project: **"Uncertainty-Aware Pneumonia Detection Using Chest X-ray Images"**.

Built with **FastAPI**, **PyTorch**, and **EfficientNet-B0 (Transfer Learning)**, featuring **Monte Carlo Dropout** uncertainty estimation and **Grad-CAM** saliency explainability.

---

## 📑 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model Checkpoint Placement](#model-checkpoint-placement)
- [Running the Server](#running-the-server)
- [Interactive API Documentation](#interactive-api-documentation)
- [API Endpoints & Schemas](#api-endpoints--schemas)
  - [1. Health Check (`GET /api/health`)](#1-health-check-get-apihealth)
  - [2. Model Information (`GET /api/model-info`)](#2-model-information-get-apimodel-info)
  - [3. Standard Prediction (`POST /api/predict`)](#3-standard-prediction-post-apipredict)
  - [4. Uncertainty Estimation (`POST /api/predict-with-uncertainty`)](#4-uncertainty-estimation-post-apipredict-with-uncertainty)
  - [5. Grad-CAM Explainability (`POST /api/explain`)](#5-grad-cam-explainability-post-apiexplain)
  - [6. Combined Analysis (`POST /api/analyze`)](#6-combined-analysis-post-apianalyze)
- [Testing](#testing)
- [Connecting Frontend (React / Streamlit)](#connecting-frontend-react--streamlit)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```text
backend/
├── app/
│   ├── __init__.py           # Application package
│   ├── main.py               # FastAPI app, lifespan, CORS, and exception handlers
│   ├── config.py             # Settings, hyperparameters, and path resolution
│   ├── schemas.py            # Pydantic request and response models
│   ├── model_loader.py       # EfficientNetB0Classifier and ModelManager singleton
│   ├── preprocessing.py      # Image validation, security checks, and eval transforms
│   ├── inference.py          # Deterministic classification inference
│   ├── uncertainty.py        # Monte Carlo Dropout sampling and entropy metrics
│   ├── explainability.py     # Native PyTorch Grad-CAM (targeting conv_head)
│   ├── utils.py              # Base64 image encoders and custom exceptions
│   └── routers/
│       ├── __init__.py
│       ├── health.py         # /health and /model-info endpoints
│       └── predictions.py    # /predict, /predict-with-uncertainty, /explain, /analyze
├── models/
│   ├── .gitkeep
│   └── efficientnet_b0_best.pth  (Trained checkpoint, ~17 MB)
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Lightweight mock models and in-memory test images
│   ├── test_health.py        # Health and model-info unit tests
│   └── test_predictions.py   # Prediction, uncertainty, and Grad-CAM tests
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore
└── README.md
```

### Key Technical Decisions
1. **Source of Truth:** Architecture matches `ChestXRay_Pneumonia_Detection_v2.ipynb` Cell 7: `timm.create_model('efficientnet_b0')` with custom classification head `nn.Sequential(BatchNorm1d, Dropout(0.35), Linear(1280, 512), SiLU, Dropout(0.175), Linear(512, 2))`.
2. **One-Time Model Loading:** The checkpoint is loaded into memory once during FastAPI startup. Individual requests never reload disk weights.
3. **Correct Monte Carlo Dropout:** Dropout layers are selectively activated (`m.train()`) while Batch Normalization layers are kept in evaluation mode (`model.eval()`). This prevents corruption of learned batch statistics while generating stochastic forward passes.
4. **Grad-CAM on `backbone.conv_head`:** Targets the exact 1x1 convolution head from Cell 10 of the notebook, calculating gradient-weighted class activations and producing base64-encoded visual overlays.
5. **No Medical Hallucination:** The model outputs research metrics. Because no clinical threshold was validated in the notebook, uncertainty review recommendations are marked `EXPERIMENTAL_UNREVIEWED` by default unless configured by the user.

---

## Prerequisites
- **Python**: Version 3.9, 3.10, or 3.11 (64-bit)
- **PyTorch**: 2.0+ (CPU or CUDA 11.8/12.1)
- **Trained Weights**: `efficientnet_b0_best.pth` (present in project root or `backend/models/`)

---

## Environment Setup

### 1. Create a Python Virtual Environment
Navigate to the `backend` folder:
```bash
cd backend
python -m venv venv
```

Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create `.env` Configuration (Optional)
Copy the example configuration:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

---

## Model Checkpoint Placement

The backend automatically searches for the checkpoint in:
1. The path configured in `MODEL_CHECKPOINT_PATH` in `.env`
2. `backend/models/efficientnet_b0_best.pth`
3. `../efficientnet_b0_best.pth` (project root)
4. `d:/pneumopred/efficientnet_b0_best.pth`

You can also copy the trained weights into `backend/models/`:
```bash
# From the project root:
cp efficientnet_b0_best.pth backend/models/
```

> **Note for Colab Users:** If you train a new checkpoint in Google Colab, download `efficientnet_b0_best.pth` and place it into `backend/models/`.

---

## Running the Server

Make sure your working directory is `backend/` and run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Console output will confirm the loaded checkpoint and device:
```text
INFO:     Initializing Uncertainty-Aware Pneumonia Detection Backend...
INFO:     Found checkpoint at: D:\pneumopred\efficientnet_b0_best.pth
INFO:     Successfully loaded model checkpoint and initialized in evaluation mode.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Interactive API Documentation

Once the server is running, open your browser:
- **Swagger UI (Interactive):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc (Alternative Reference):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## API Endpoints & Schemas

### 1. Health Check (`GET /api/health`)
Verifies service status, compute device, and whether the model checkpoint is loaded.

**Request:**
```bash
curl -X GET http://127.0.0.1:8000/api/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_architecture": "EfficientNet-B0 (timm)",
  "device": "cpu",
  "checkpoint_path": "D:\\pneumopred\\efficientnet_b0_best.pth",
  "version": "1.0.0"
}
```

---

### 2. Model Information (`GET /api/model-info`)
Returns model architecture, target classes, normalization parameters, and verified features.

**Request:**
```bash
curl -X GET http://127.0.0.1:8000/api/model-info
```

**Response (200 OK):**
```json
{
  "model_name": "EfficientNet-B0 (timm)",
  "architecture": "EfficientNet-B0 backbone with custom classification head: BatchNorm1d(1280) -> Dropout(0.35) -> Linear(1280, 512) -> SiLU -> Dropout(0.175) -> Linear(512, 2)",
  "supported_classes": ["NORMAL", "PNEUMONIA"],
  "class_to_idx": {
    "NORMAL": 0,
    "PNEUMONIA": 1
  },
  "input_dimensions": {
    "width": 224,
    "height": 224,
    "channels": 3,
    "color_space": "RGB"
  },
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "checkpoint_status": {
    "is_loaded": true,
    "path": "D:\\pneumopred\\efficientnet_b0_best.pth",
    "file_exists": true,
    "file_size_mb": 16.83,
    "load_error": null
  },
  "uncertainty_estimation_available": true,
  "gradcam_explainability_available": true,
  "device": "cpu"
}
```

---

### 3. Standard Prediction (`POST /api/predict`)
Accepts a chest X-ray image (`image/jpeg` or `image/png`) via multipart form-data.

**cURL Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@/path/to/chest_xray.jpeg"
```

**Python Request:**
```python
import requests

with open("chest_xray.jpeg", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/api/predict",
        files={"file": ("chest_xray.jpeg", f, "image/jpeg")}
    )
print(response.json())
```

**Response (200 OK):**
```json
{
  "success": true,
  "predicted_class": "PNEUMONIA",
  "predicted_index": 1,
  "confidence": 0.9412,
  "probabilities": {
    "NORMAL": 0.0588,
    "PNEUMONIA": 0.9412
  },
  "model": "EfficientNet-B0 (timm)",
  "device": "cpu",
  "inference_time_ms": 32.45,
  "message": "Research prototype output; not a clinical diagnosis."
}
```

---

### 4. Uncertainty Estimation (`POST /api/predict-with-uncertainty`)
Performs $T$ stochastic Monte Carlo Dropout forward passes to compute predictive entropy, aleatoric expected entropy, and epistemic mutual information.

**Query Parameters:**
- `num_samples` *(optional, int, default=20, min=2, max=100)*: Number of stochastic passes.
- `threshold` *(optional, float, range=[0.0, 1.0])*: Experimental entropy threshold in bits.

**cURL Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/predict-with-uncertainty?num_samples=25&threshold=0.45" \
  -F "file=@/path/to/chest_xray.jpeg"
```

**Response (200 OK):**
```json
{
  "success": true,
  "predicted_class": "PNEUMONIA",
  "predicted_index": 1,
  "mean_probabilities": {
    "NORMAL": 0.1235,
    "PNEUMONIA": 0.8765
  },
  "uncertainty": {
    "predictive_entropy": 0.5372,
    "expected_entropy": 0.3814,
    "mutual_information": 0.1558,
    "predictive_variance": {
      "NORMAL": 0.004218,
      "PNEUMONIA": 0.004218
    },
    "mean_variance": 0.004218,
    "standard_deviation": {
      "NORMAL": 0.064946,
      "PNEUMONIA": 0.064946
    },
    "entropy_unit": "bits (log2)",
    "num_stochastic_passes": 25
  },
  "decision_support": {
    "review_recommendation": "FLAGGED_FOR_HUMAN_REVIEW",
    "threshold_configured": true,
    "threshold_value": 0.45,
    "note": "Predictive entropy (0.5372 bits) exceeds configured research threshold (0.4500 bits). Second opinion or radiologist review recommended."
  },
  "model": "EfficientNet-B0 (timm)",
  "device": "cpu",
  "inference_time_ms": 184.21,
  "message": "Research prototype output; not a clinical diagnosis."
}
```

---

### 5. Grad-CAM Explainability (`POST /api/explain`)
Generates spatial saliency maps on `backbone.conv_head` and returns Base64 PNGs.

**Query Parameters:**
- `target_class` *(optional, int, 0=NORMAL, 1=PNEUMONIA)*: Target class to explain.
- `alpha` *(optional, float, default=0.45)*: Heatmap transparency overlay weight.

**cURL Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/explain?alpha=0.5" \
  -F "file=@/path/to/chest_xray.jpeg"
```

**Response (200 OK):**
```json
{
  "success": true,
  "predicted_class": "PNEUMONIA",
  "target_class_explained": "PNEUMONIA",
  "target_class_index": 1,
  "target_layer": "backbone.conv_head",
  "overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "inference_time_ms": 68.32,
  "message": "Grad-CAM visualization highlights regions influencing model prediction. Research prototype output; does not prove clinical reasoning."
}
```

---

### 6. Combined Analysis (`POST /api/analyze`)
Single comprehensive endpoint combining classification, uncertainty, and Grad-CAM for frontend dashboards.

**Query Parameters:**
- `include_uncertainty` *(bool, default=true)*
- `include_explainability` *(bool, default=true)*
- `num_samples` *(int, default=20)*
- `uncertainty_threshold` *(optional, float)*
- `target_class` *(optional, int)*

**cURL Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/analyze?include_uncertainty=true&include_explainability=true&num_samples=20" \
  -F "file=@/path/to/chest_xray.jpeg"
```

---

## Testing

The backend includes a comprehensive test suite using **pytest** and FastAPI's **TestClient**. Unit tests run in-memory on CPU using lightweight mock models and synthetic images, so tests can run without GPU or network dependencies.

Run all tests:
```bash
cd backend
pytest -v
```

---

## Connecting Frontend (React / Streamlit)

### Example: Streamlit
```python
import streamlit as st
import requests

st.title("🫁 Uncertainty-Aware Pneumonia Detection")
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("Run Full Analysis"):
    with st.spinner("Analyzing image..."):
        res = requests.post(
            "http://127.0.0.1:8000/api/analyze",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        ).json()

        st.subheader(f"Prediction: {res['predicted_class']} ({res['confidence']:.1%})")
        
        # Uncertainty
        unc = res.get("uncertainty", {})
        st.metric("Predictive Entropy", f"{unc.get('predictive_entropy', 0):.3f} bits")
        st.metric("Mutual Information", f"{unc.get('mutual_information', 0):.3f} bits")
        
        # Grad-CAM
        exp = res.get("explanation", {})
        if exp and "overlay_base64" in exp:
            st.image(exp["overlay_base64"], caption="Grad-CAM Saliency Overlay")
```

### Example: React / Next.js
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://127.0.0.1:8000/api/analyze", {
  method: "POST",
  body: formData,
});
const data = await response.json();
console.log("Predicted:", data.predicted_class);
// Render data.explanation.overlay_base64 in an <img src={...} /> tag
```

---

## Troubleshooting

1. **`503 Service Unavailable: Model is not loaded`**
   - Check that `efficientnet_b0_best.pth` is in `backend/models/` or in the project root.
   - Verify file permissions.
   - Inspect `/api/health` to see the exact checkpoint path and error message.

2. **`ModuleNotFoundError: No module named 'timm'`**
   - Run `pip install timm>=0.9.0` in your active virtual environment.

3. **`400 Bad Request: Unsupported file extension`**
   - Upload only valid `.jpg`, `.jpeg`, or `.png` files.

4. **CUDA Out of Memory (OOM)**
   - Set `DEVICE=cpu` in `backend/.env` or reduce `num_samples` in uncertainty requests.

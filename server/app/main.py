import time
import logging
import os
from contextlib import asynccontextmanager
from collections import Counter
from threading import Lock

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.vision.preprocessing import preprocess_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("dermatology-api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/efficientnet_quant_int8.onnx")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# HAM10000 Class Labels
DIAGNOSES = [
    "Actinic keratoses / Intraepithelial carcinoma (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vasc)",
]

# ---------------------------------------------------------------------------
# Metrics (thread-safe atomic counters)
# ---------------------------------------------------------------------------
_metrics_lock = Lock()
_metrics: dict = {
    "requests_total": 0,
    "requests_failed": 0,
    "inference_time_total_ms": 0.0,
}

def _record_request(latency_ms: float, failed: bool = False) -> None:
    with _metrics_lock:
        _metrics["requests_total"] += 1
        _metrics["inference_time_total_ms"] += latency_ms
        if failed:
            _metrics["requests_failed"] += 1

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Starting up — loading ONNX model from %s", MODEL_PATH)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # Warm-up: eliminates cold-start latency on first real request
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logger.info("Executing 3 warm-up runs…")
        for _ in range(3):
            session.run(None, {input_name: dummy})
        logger.info("Warm-up complete — service is ready.")
        app.state.session = session
        app.state.model_path = MODEL_PATH
        app.state.input_name = input_name
        app.state.input_shape = session.get_inputs()[0].shape
        app.state.providers = session.get_providers()
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        app.state.session = None

    yield  # Application runs here

    logger.info("Shutting down.")
    app.state.session = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dermatology AI Inference Service",
    version="2.0.0",
    description="Enterprise-grade skin lesion classification microservice using ONNX Runtime INT8 quantization.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: list[float]
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str

class MetricsResponse(BaseModel):
    requests_total: int
    requests_failed: int
    inference_time_avg_ms: float

class InfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_path: str
    input_shape: list
    providers: list[str]
    diagnoses: list[str]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Operations"])
async def root():
    """Root endpoint indicating service status."""
    return {
        "service": "Dermatology AI Inference Service",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/healthz", response_model=HealthResponse, tags=["Operations"])
async def health_check(request: Request):
    """Kubernetes liveness / readiness probe."""
    if request.app.state.session is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": "Model not loaded"},
        )
    return {"status": "healthy"}


@app.get("/metrics", response_model=MetricsResponse, tags=["Operations"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    with _metrics_lock:
        total = _metrics["requests_total"]
        avg = (
            _metrics["inference_time_total_ms"] / total if total > 0 else 0.0
        )
        return MetricsResponse(
            requests_total=total,
            requests_failed=_metrics["requests_failed"],
            inference_time_avg_ms=round(avg, 2),
        )


@app.get("/info", response_model=InfoResponse, tags=["Operations"])
async def info(request: Request):
    """Expose model metadata for observability."""
    if request.app.state.session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return InfoResponse(
        model_path=request.app.state.model_path,
        input_shape=list(request.app.state.input_shape),
        providers=request.app.state.providers,
        diagnoses=DIAGNOSES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Accepts a dermoscopic image (JPEG / PNG / BMP), applies the Dull Razor
    preprocessing pipeline, and returns an ONNX-based skin lesion classification.
    """
    session = request.app.state.session
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # MIME type validation
    valid_mime_types = {"image/jpeg", "image/png", "image/bmp"}
    if file.content_type not in valid_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: JPEG, PNG, BMP.",
        )

    # File size guard
    if getattr(file, 'size', 0) and getattr(file, 'size', 0) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file.size // 1024} KB). Maximum allowed: 10 MB.",
        )
    contents = await file.read()

    start_time = time.perf_counter()
    try:
        input_tensor = preprocess_image(contents)
        outputs = session.run(None, {request.app.state.input_name: input_tensor})

        logits = outputs[0][0]
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        if predicted_idx < len(DIAGNOSES):
            diagnosis = DIAGNOSES[predicted_idx]
        else:
            diagnosis = f"Unknown Class (Index {predicted_idx})"

        latency_ms = (time.perf_counter() - start_time) * 1000
        _record_request(latency_ms)

        logger.info(
            "Prediction: %s (conf=%.2f%%) in %.1f ms",
            diagnosis,
            confidence * 100,
            latency_ms,
        )

        return PredictionResponse(
            diagnosis=diagnosis,
            confidence=round(confidence, 4),
            probabilities=probs.tolist(),
            processing_time_ms=round(latency_ms, 2),
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        _record_request(latency_ms, failed=True)
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

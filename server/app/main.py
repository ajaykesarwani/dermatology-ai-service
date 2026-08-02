import time
import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
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
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/resnet18_quant_int8.onnx")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_CHUNK_SIZE = 64 * 1024                 # 64 KB read chunks for streaming size guard

# Fix #7 — CORS origins driven by environment variable; defaults to restrictive empty list.
# Set ALLOWED_ORIGINS="*" in dev, or comma-separated origins in production.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins and _raw_origins != "*"
    else (["*"] if _raw_origins == "*" else [])
)

# Fix #8 — Optional API key protection for the /metrics endpoint.
# Leave METRICS_API_KEY unset (or empty) to disable authentication in development.
_METRICS_API_KEY = os.environ.get("METRICS_API_KEY", "")
_metrics_key_header = APIKeyHeader(name="X-Metrics-Key", auto_error=False)

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
# Helpers
# ---------------------------------------------------------------------------
async def _read_with_size_limit(file: UploadFile) -> bytes:
    """
    Fix #4 — Stream-reads the upload in 64 KB chunks and enforces the size
    limit on actual bytes received.  The old approach relied on file.size which
    is None when the client omits a Content-Length header, silently bypassing
    the guard entirely.
    """
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        received += len(chunk)
        if received > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({received // 1024} KB received so far). "
                    f"Maximum allowed: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _require_metrics_key(
    api_key: str | None = Security(_metrics_key_header),
) -> None:
    """
    M1 FIX — FastAPI dependency (async function) for protecting the /metrics
    endpoint with an API key header.  Previously this was a regular (non-async)
    function used as a default parameter value, which is non-standard and
    invisible to FastAPI's OpenAPI schema generator.

    Usage: inject with `dependencies=[Depends(_require_metrics_key)]` on the route.
    When METRICS_API_KEY env var is empty, the check is skipped (dev mode).
    """
    if _METRICS_API_KEY and api_key != _METRICS_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing metrics API key.")


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
    version="2.1.0",
    description="Enterprise-grade skin lesion classification microservice using ONNX Runtime INT8 quantization.",
    lifespan=lifespan,
)

# Fix #7 — CORS is now controlled via ALLOWED_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    """Root endpoint serving the Web UI."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {
        "service": "Dermatology AI Inference Service",
        "status": "online",
        "documentation": "/docs",
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
async def metrics(request: Request, _: None = Security(_require_metrics_key)):
    """
    Prometheus-compatible metrics endpoint.
    Protected by X-Metrics-Key header when METRICS_API_KEY env var is set.
    M1 FIX — dependency injected via Security() directly in route signature
    (standard FastAPI pattern; shows in OpenAPI docs and is statically analyzable).
    """
    with _metrics_lock:
        total = _metrics["requests_total"]
        avg = _metrics["inference_time_total_ms"] / total if total > 0 else 0.0
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

    # Fix #4 — streaming size guard that works even when Content-Length is absent
    contents = await _read_with_size_limit(file)

    start_time = time.perf_counter()
    try:
        input_tensor = preprocess_image(contents)
        outputs = session.run(None, {request.app.state.input_name: input_tensor})

        logits = outputs[0][0]
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        diagnosis = (
            DIAGNOSES[predicted_idx]
            if predicted_idx < len(DIAGNOSES)
            else f"Unknown Class (Index {predicted_idx})"
        )

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

    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        _record_request(latency_ms, failed=True)
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

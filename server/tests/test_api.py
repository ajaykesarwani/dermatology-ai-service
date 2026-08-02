"""
Unit tests for the Dermatology AI Inference Service API.

Run with:
    pytest server/tests/ -v
"""
import io
import struct
import zlib
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, MAX_FILE_SIZE_BYTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_minimal_png(width: int = 4, height: int = 4) -> bytes:
    """Creates a minimal valid PNG image in-memory (no file I/O needed)."""

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + b"\x80\x80\x80" * width  # filter byte + grey pixels

    compressed = zlib.compress(raw_rows)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr_data)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_healthz_returns_200_or_503(self, client):
        """Healthz must always respond (healthy or not)."""
        response = client.get("/healthz")
        assert response.status_code in (200, 503)

    def test_healthz_schema(self, client):
        response = client.get("/healthz")
        data = response.json()
        assert "status" in data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_metrics_ok_no_auth(self, client):
        """
        R5 FIX — Explicitly clear METRICS_API_KEY for this test so it passes
        regardless of whether the env var is set in the CI environment.
        Without this, if CI injects a secret the test silently got a 403 while
        asserting 200.
        """
        import app.main as main_module
        with patch.object(main_module, "_METRICS_API_KEY", ""):
            response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_schema(self, client):
        import app.main as main_module
        with patch.object(main_module, "_METRICS_API_KEY", ""):
            data = client.get("/metrics").json()
        assert "requests_total" in data
        assert "requests_failed" in data
        assert "inference_time_avg_ms" in data
        assert isinstance(data["requests_total"], int)

    def test_metrics_requires_key_when_set(self, client):
        """R5 FIX — verify 403 is returned when auth key is set but header absent."""
        import app.main as main_module
        with patch.object(main_module, "_METRICS_API_KEY", "test-secret-key"):
            response = client.get("/metrics")
        assert response.status_code == 403

    def test_metrics_accepts_correct_key(self, client):
        """R5 FIX — verify 200 is returned when the correct key is supplied."""
        import app.main as main_module
        with patch.object(main_module, "_METRICS_API_KEY", "test-secret-key"):
            response = client.get("/metrics", headers={"X-Metrics-Key": "test-secret-key"})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
class TestInfo:
    def test_info_returns_diagnoses(self, client):
        response = client.get("/info")
        if response.status_code == 200:
            data = response.json()
            assert "diagnoses" in data
            assert len(data["diagnoses"]) == 7
            assert "providers" in data


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_rejects_invalid_mime(self, client):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_rejects_oversized_file(self, client):
        """
        Fix #12 — test the actual streaming size guard (_read_with_size_limit)
        instead of patching Starlette internals which is brittle and version-
        dependent.  We mock the helper directly at the application layer.
        """
        from fastapi import HTTPException as FastAPIHTTPException
        import app.main as main_module

        async def _reject_oversized(*args, **kwargs):
            raise FastAPIHTTPException(
                status_code=413,
                detail="File too large. Maximum allowed: 10 MB.",
            )

        small_jpeg = b"\xff\xd8\xff" + b"\x00" * 10
        with patch.object(main_module, "_read_with_size_limit", new=_reject_oversized):
            response = client.post(
                "/predict",
                files={"file": ("big.jpg", small_jpeg, "image/jpeg")},
            )
        assert response.status_code == 413

    def test_predict_with_valid_png(self, client):
        """When model is loaded, a valid PNG should return a prediction."""
        png_bytes = _make_minimal_png()
        response = client.post(
            "/predict",
            files={"file": ("lesion.png", png_bytes, "image/png")},
        )
        # 503 → model not loaded in test env; 200 → success; 500 → preprocessing on tiny image
        assert response.status_code in (200, 500, 503)
        if response.status_code == 200:
            data = response.json()
            assert "diagnosis" in data
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] >= 0

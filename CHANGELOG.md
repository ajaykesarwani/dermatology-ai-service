# DermDiagnostic AI — Changelog

All notable changes to this project are documented here.

---

## v2.1.0 — Production Hardening & Training Pipeline

### Added
- **Weighted training pipeline**: `WeightedRandomSampler` + `CrossEntropyLoss(weight=...)` to handle HAM10000 class imbalance (~67% melanocytic nevi)
- **Validation split**: 15% stratified hold-out per epoch; best checkpoint saved by validation loss
- **LR scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=2) added to training loop
- **CORS environment control**: `ALLOWED_ORIGINS` env var drives CORS policy; defaults to restrictive empty list
- **Metrics authentication**: `X-Metrics-Key` header guard on `/metrics`; key stored as platform secret
- **Streaming file size guard**: `_read_with_size_limit()` enforces 10 MB cap on actual bytes received (not header)
- **Model warm-up**: 3 ONNX warm-up runs on startup to eliminate cold-start latency
- **Triton support**: Full Triton Inference Server configuration with TensorRT FP16, dynamic batching, CPU/GPU fallback
- **AWS ECS Fargate**: One-command deployment with ECR, ALB, CloudWatch, circuit breaker, and Terraform IaC
- **Render deployment**: `render.yaml` with correct PORT injection support
- **Kubernetes manifest**: Deployment + Service with liveness/readiness probes and Secrets Manager integration
- **GitHub Actions CI**: Ruff lint, pytest (with PYTHONPATH), docker-compose validation, Docker build
- **.dockerignore**: Prevents venv/ and large binary files from bloating the Docker build context
- **Reproducible training**: Fixed `np.random.seed(42)` for consistent train/val splits

### Fixed
- `GaussianBlur` sigma argument was incorrectly using `cv2.BORDER_DEFAULT` (=4) instead of `0` (auto-calculate)
- ONNX quantization used `QUInt8` instead of `QInt8`, mismatching the `*_int8*` filename convention
- Triton client skipped the Dull Razor hair-removal preprocessing step, degrading inference accuracy
- Docker `ENV` inline comments corrupted variable values (Docker treats `#` as part of the value)
- Terraform `main.tf` used the old `efficientnet_quant_int8.onnx` model path
- CI `pytest` lacked `PYTHONPATH=server`, causing `ModuleNotFoundError` on every CI run

### Changed
- All model filenames updated from `efficientnet_*` → `resnet18_*` to reflect the actual architecture
- `pretrained=True` replaced with `weights=ResNet18_Weights.IMAGENET1K_V1` (non-deprecated API)
- Training epochs increased from 5 → 10
- `run_epoch()` optimizer parameter made `Optional`; validation calls no longer pass it
- Class weight computation moved to a single O(n) pass over raw HF dataset (no image transforms)

---

## v2.0.0 — Initial Architecture

- FastAPI inference microservice with ONNX Runtime INT8 quantization
- Dull Razor hair-removal preprocessing pipeline
- ResNet-18 fine-tuned on HAM10000 (7-class skin lesion classification)
- C# WPF desktop client with dark UI, confidence bar, and color-coded risk indicator
- Docker Compose deployment

"""
Benchmark script comparing PyTorch eager mode vs ONNX Runtime (FP32 & INT8).

Usage:
    python benchmark.py
    python benchmark.py --onnx-path resnet18.onnx --quant-path resnet18_quant_int8.onnx --iters 100
"""
import time
import os
import argparse

import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort


def benchmark_pytorch(model, dummy_input, num_iterations: int = 100, device: str = "cpu"):
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    print(f"\nBenchmarking PyTorch Eager Mode ({device.upper()})…")
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            model(dummy_input)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / num_iterations) * 1000
    print(f"  Average latency: {avg_latency_ms:.2f} ms")
    return avg_latency_ms


def benchmark_onnx(
    model_path: str,
    dummy_input_np: np.ndarray,
    num_iterations: int = 100,
    provider: str = "CPUExecutionProvider",
):
    if not os.path.exists(model_path):
        print(f"  Model {model_path} not found — skipping.")
        return None

    print(f"\nBenchmarking ONNX Runtime ({provider})  [{model_path}]…")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, session_options, providers=[provider])
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: dummy_input_np})

    start = time.perf_counter()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy_input_np})
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / num_iterations) * 1000
    print(f"  Average latency: {avg_latency_ms:.2f} ms")
    return avg_latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ResNet-18 inference latency")
    # Fix #1 — default paths now reflect the actual model architecture (resnet18, not efficientnet)
    parser.add_argument("--onnx-path",  type=str, default="resnet18.onnx")
    parser.add_argument("--quant-path", type=str, default="resnet18_quant_int8.onnx")
    parser.add_argument("--iters",      type=int, default=50)
    args = parser.parse_args()

    print("=== Dermatology AI — Inference Benchmark ===")

    # Fix #2 — use non-deprecated weights= API
    model = models.resnet18(weights=None)   # random weights are sufficient for latency benchmarking
    model.eval()

    dummy_torch = torch.randn(1, 3, 224, 224)
    dummy_np    = dummy_torch.numpy()

    # 1. PyTorch Baseline (CPU)
    benchmark_pytorch(model, dummy_torch, num_iterations=args.iters, device="cpu")

    # 2. PyTorch Baseline (GPU, if available)
    if torch.cuda.is_available():
        benchmark_pytorch(model, dummy_torch, num_iterations=args.iters, device="cuda")

    # 3. ONNX Runtime FP32 (CPU)
    benchmark_onnx(args.onnx_path, dummy_np, num_iterations=args.iters, provider="CPUExecutionProvider")

    # 4. ONNX Runtime FP32 (GPU, if available)
    if "CUDAExecutionProvider" in ort.get_available_providers():
        benchmark_onnx(args.onnx_path, dummy_np, num_iterations=args.iters, provider="CUDAExecutionProvider")

    # 5. ONNX Runtime INT8 Quantized (CPU)
    benchmark_onnx(args.quant_path, dummy_np, num_iterations=args.iters, provider="CPUExecutionProvider")

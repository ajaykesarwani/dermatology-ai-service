import time
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
import argparse
import os

def benchmark_pytorch(model, dummy_input, num_iterations=100, device='cpu'):
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    print(f"\nBenchmarking PyTorch Eager Mode ({device.upper()})...")
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            model(dummy_input)
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / num_iterations) * 1000
    print(f"Average Latency: {avg_latency:.2f} ms")
    return avg_latency

def benchmark_onnx(model_path, dummy_input_np, num_iterations=100, provider='CPUExecutionProvider'):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return None
        
    print(f"\nBenchmarking ONNX Runtime ({provider}, {model_path})...")
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, session_options, providers=[provider])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input_np})
        
    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: dummy_input_np})
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / num_iterations) * 1000
    print(f"Average Latency: {avg_latency:.2f} ms")
    return avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model latency")
    parser.add_argument("--onnx-path", type=str, default="efficientnet.onnx")
    parser.add_argument("--quant-path", type=str, default="efficientnet_quant_int8.onnx")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print("=== Medical CV Inference Benchmark ===")
    
    # 1. PyTorch Baseline
    model = models.resnet18(pretrained=False)
    model.eval()
    dummy_input_torch = torch.randn(1, 3, 224, 224)
    dummy_input_np = dummy_input_torch.numpy()
    
    benchmark_pytorch(model, dummy_input_torch, num_iterations=args.iters, device='cpu')
    if torch.cuda.is_available():
        benchmark_pytorch(model, dummy_input_torch, num_iterations=args.iters, device='cuda')
        
    # 2. ONNX Runtime FP32
    benchmark_onnx(args.onnx_path, dummy_input_np, num_iterations=args.iters, provider='CPUExecutionProvider')
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        benchmark_onnx(args.onnx_path, dummy_input_np, num_iterations=args.iters, provider='CUDAExecutionProvider')
        
    # 3. ONNX Runtime INT8 Quantized
    benchmark_onnx(args.quant_path, dummy_input_np, num_iterations=args.iters, provider='CPUExecutionProvider')

import torch
import torchvision.models as models
import onnx
import argparse

def export_model(output_path: str = "skin_lesion_model.onnx", quantize: bool = True):
    print("Loading pretrained ResNet18 (simulating a skin lesion classification model)...")
    model = models.resnet18(pretrained=True)
    model.eval()

    # Dummy input for medical image (e.g., 224x224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Export successful.")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
            quant_path = output_path.replace(".onnx", "_quant_int8.onnx")
            preprocessed_path = output_path.replace(".onnx", "_preprocessed.onnx")
            
            print("Running ONNX shape inference preprocessing...")
            shape_inference.quant_pre_process(output_path, preprocessed_path, skip_optimization=False)
            
            print(f"Quantizing model to {quant_path} (INT8)...")
            quantize_dynamic(preprocessed_path, quant_path, weight_type=QuantType.QUInt8)
            print("Quantization successful.")
        except ImportError:
            print("onnxruntime not installed. Skipping quantization.")
        except Exception as e:
            print(f"Quantization skipped due to shape inference error (common in PyTorch -> ONNX exports without pre-processing): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--output", type=str, default="efficientnet.onnx", help="Output ONNX file path")
    parser.add_argument("--no-quantize", action="store_true", help="Disable INT8 quantization")
    args = parser.parse_args()
    
    export_model(args.output, not args.no_quantize)

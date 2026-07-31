import torch
import subprocess
import sys
import os

print("Installing required Hugging Face libraries...")
try:
    import transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "huggingface_hub"])

from transformers import AutoModelForImageClassification

def download_and_export():
    # Hugging Face model trained on HAM10000 Skin Lesion Dataset
    model_name = "PREMAADC/vit-base-ham10000"
    print(f"Downloading pre-trained dermatology model: {model_name}...")
    
    # Load model from Hugging Face Hub
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    
    # Dummy input for medical image (e.g., 224x224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    output_path = "efficientnet.onnx"
    print(f"Exporting model to {output_path}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("Export successful.")
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
        quant_path = "efficientnet_quant_int8.onnx"
        preprocessed_path = "efficientnet_preprocessed.onnx"
        
        print("Running ONNX shape inference preprocessing...")
        shape_inference.quant_pre_process(output_path, preprocessed_path, skip_optimization=False)
        
        print(f"Quantizing model to {quant_path} (INT8)...")
        quantize_dynamic(preprocessed_path, quant_path, weight_type=QuantType.QUInt8)
        print("Quantization successful.")
    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    download_and_export()

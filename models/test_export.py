import torch
from transformers import AutoModelForImageClassification
import sys

def main():
    model_name = "kusumrss/ham10000-efficientnet-b2"
    print(f"Downloading model: {model_name}...")
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        output_path = "test_export.onnx"
        
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
        print("Export successful!")
        
        # Print id2label
        print("Classes:", model.config.id2label)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

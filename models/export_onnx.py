"""
Export a trained ResNet-18 model to ONNX (FP32) and optionally quantize to INT8.

Usage:
    python export_onnx.py                          # exports resnet18.onnx + quantized
    python export_onnx.py --output mymodel.onnx   # custom output path
    python export_onnx.py --no-quantize            # skip quantization
"""
import os
import argparse
import torch
import torchvision.models as models
import onnx


def export_model(output_path: str = "resnet18.onnx", quantize: bool = True):
    # Fix #1 — model is ResNet-18, not EfficientNet; naming reflects reality
    print("Loading ResNet-18 adapted for 7 HAM10000 skin-lesion classes…")

    # Fix #2 — use non-deprecated weights= API (pretrained=True was removed in torchvision 0.15)
    model = models.resnet18(weights=None)           # architecture only; weights loaded below
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)

    weights_path = os.path.join(os.path.dirname(__file__), "resnet18_ham10000.pth")
    if os.path.exists(weights_path):
        print(f"✅ Found trained weights at {weights_path}. Loading…")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(
            f"⚠️  WARNING: No trained weights found at {weights_path}.\n"
            "   Exporting an untrained model — run train.py first for a clinically useful model."
        )

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting FP32 ONNX model to {output_path}…")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify the exported graph is valid
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX export verified successfully.")

    if not quantize:
        return

    try:
        # Fix #13 — use explicit sub-module imports that are stable across onnxruntime versions
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process

        preprocessed_path = output_path.replace(".onnx", "_preprocessed.onnx")
        # Fix #1 — output file name reflects actual model architecture
        quant_path = output_path.replace(".onnx", "_quant_int8.onnx")

        print("Running ONNX shape-inference pre-processing…")
        quant_pre_process(output_path, preprocessed_path, skip_optimization=False)

        # Fix #6 — use QInt8 (signed) which matches the "*_int8*" naming convention
        print(f"Quantizing to signed INT8 → {quant_path}…")
        quantize_dynamic(preprocessed_path, quant_path, weight_type=QuantType.QInt8)
        print("✅ INT8 quantization complete.")

    except ImportError:
        print("onnxruntime not installed. Skipping quantization.")
    except Exception as exc:
        print(f"Quantization failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ResNet-18 to ONNX")
    parser.add_argument(
        "--output", type=str, default="resnet18.onnx",
        help="Output ONNX file path (default: resnet18.onnx)",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Disable INT8 quantization step",
    )
    args = parser.parse_args()
    export_model(args.output, not args.no_quantize)

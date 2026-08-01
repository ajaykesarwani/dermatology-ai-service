#!/usr/bin/env python3
"""
NVIDIA Triton Inference Server — Client Example
================================================
Demonstrates how to query the Triton-hosted skin lesion classifier
using the tritonclient HTTP library.

Usage:
    pip install tritonclient[http] pillow numpy
    python deployment/triton/triton_client.py --image path/to/lesion.jpg
"""
import argparse
import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

# HAM10000 class labels (must match Triton model config output order)
DIAGNOSES = [
    "Actinic keratoses / Intraepithelial carcinoma (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic nevi (nv)",
    "Vascular lesions (vasc)",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess a dermoscopic image for Triton inference."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0          # [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD              # ImageNet normalize
    arr = np.transpose(arr, (2, 0, 1))                      # HWC → CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)   # add batch dim


def infer(server_url: str, image_path: str) -> None:
    client = httpclient.InferenceServerClient(url=server_url)

    # Check server health
    if not client.is_server_live():
        print(f"ERROR: Triton server at {server_url} is not live.")
        sys.exit(1)

    input_data = preprocess(image_path)

    # Build Triton input/output descriptors
    infer_input = httpclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    infer_output = httpclient.InferRequestedOutput("output")

    # Run inference
    response = client.infer(
        model_name="skin_lesion_classifier",
        inputs=[infer_input],
        outputs=[infer_output],
    )

    logits = response.as_numpy("output")[0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    predicted_idx = int(np.argmax(probs))
    print(f"\n{'='*55}")
    print(f"  Diagnosis : {DIAGNOSES[predicted_idx]}")
    print(f"  Confidence: {probs[predicted_idx]*100:.2f}%")
    print(f"\n  Full probability distribution:")
    for label, prob in zip(DIAGNOSES, probs):
        bar = "█" * int(prob * 30)
        print(f"    {label[:45]:<45} {prob*100:5.1f}% {bar}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Inference Client")
    parser.add_argument("--image", required=True, help="Path to dermoscopic image")
    parser.add_argument("--server", default="localhost:8001", help="Triton HTTP server address")
    args = parser.parse_args()
    infer(args.server, args.image)

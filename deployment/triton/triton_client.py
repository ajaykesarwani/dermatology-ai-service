#!/usr/bin/env python3
"""
NVIDIA Triton Inference Server — Client Example
================================================
Demonstrates how to query the Triton-hosted skin lesion classifier
using the tritonclient HTTP library.

Usage:
    pip install tritonclient[http] opencv-python-headless numpy
    python deployment/triton/triton_client.py --image path/to/lesion.jpg
"""
import argparse
import sys

import cv2
import numpy as np
import tritonclient.http as httpclient

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


# ---------------------------------------------------------------------------
# Preprocessing — must be identical to server/app/vision/preprocessing.py
# ---------------------------------------------------------------------------
def _dull_razor_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    M2 FIX — Dull Razor hair-removal algorithm.
    The FastAPI server applies this step before every inference; the Triton
    client previously skipped it entirely, degrading accuracy because the model
    was trained on hair-removed images.

    1. Greyscale + Black-Hat morphological filter to detect hair fibers.
    2. Gaussian blur + binary threshold to create a hair mask.
    3. Telea inpainting to restore skin texture under the hair.
    """
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    blurred   = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask   = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    mask      = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)


def preprocess(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline — mirrors server/app/vision/preprocessing.py.

    Steps:
      1. Load image with OpenCV (BGR).
      2. Dull Razor hair removal (M2 FIX — was missing before).
      3. Resize to 224 × 224.
      4. BGR → RGB.
      5. Normalise to [0, 1].
      6. ImageNet mean/std normalisation.
      7. HWC → CHW + batch dimension.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = _dull_razor_hair_removal(img)                        # M2 FIX
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)      # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer(server_url: str, image_path: str) -> None:
    client = httpclient.InferenceServerClient(url=server_url)

    if not client.is_server_live():
        print(f"ERROR: Triton server at {server_url} is not live.")
        sys.exit(1)

    input_data = preprocess(image_path)

    infer_input = httpclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    infer_output = httpclient.InferRequestedOutput("output")

    response = client.infer(
        model_name="skin_lesion_classifier",
        inputs=[infer_input],
        outputs=[infer_output],
    )

    logits      = response.as_numpy("output")[0]
    exp_logits  = np.exp(logits - np.max(logits))
    probs       = exp_logits / np.sum(exp_logits)

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
    parser.add_argument("--image",  required=True,               help="Path to dermoscopic image")
    parser.add_argument("--server", default="localhost:8001",     help="Triton HTTP server address")
    args = parser.parse_args()
    infer(args.server, args.image)

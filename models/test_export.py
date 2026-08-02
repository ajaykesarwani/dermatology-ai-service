# ⛔ DEPRECATED — This file is an orphaned R&D scratch script and is NOT
# part of the active model pipeline.
#
# What it was: An early experiment that downloaded a HuggingFace EfficientNet-B2
# model ('kusumrss/ham10000-efficientnet-b2') and exported it to ONNX using
# opset 14.  It was superseded by:
#
#   models/export_onnx.py  — the authoritative export script (ResNet-18, opset 18)
#   models/train.py        — the authoritative training script
#
# The `transformers` library this script requires is not in requirements.txt
# and must not be installed in the inference container.
#
# Do NOT run this script.  It will be removed in a future cleanup.
raise SystemExit(
    "This script is deprecated. Use models/export_onnx.py instead."
)

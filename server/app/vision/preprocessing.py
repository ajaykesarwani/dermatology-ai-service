import cv2
import numpy as np


def dull_razor_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    Implements the Dull Razor algorithm to remove hair from dermatoscopic images.

    Steps:
    1. Grayscale conversion.
    2. Black-Hat morphological filtering (extracts hair-like dark fibers).
    3. Hair mask creation via Gaussian blur + thresholding + dilation.
    4. Telea image inpainting to restore skin texture underneath the hair.

    Args:
        image: BGR numpy array representing the original image.

    Returns:
        BGR numpy array with hairs removed.
    """
    # 1. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Black-Hat morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Hair mask creation
    # Fix #5 — third argument to GaussianBlur is sigmaX (float), NOT a border-type
    # constant.  cv2.BORDER_DEFAULT == 4, which was accidentally used as sigma=4.
    # Using sigma=0 instructs OpenCV to auto-calculate sigma from the kernel size,
    # which is the standard, correct approach.
    blurred = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, hair_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # Dilate the mask slightly to ensure hair edges are fully covered
    kernel_dilate = np.ones((3, 3), np.uint8)
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)

    # 4. Telea inpainting — restores skin texture based on neighbouring pixels
    restored_image = cv2.inpaint(image, hair_mask, 6, cv2.INPAINT_TELEA)

    return restored_image


def preprocess_image(image_bytes: bytes, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline for incoming dermoscopic images.

    Steps:
      1. Decode raw bytes to a BGR numpy array.
      2. Dull Razor hair removal.
      3. Resize to model input size (224 × 224 by default).
      4. BGR → RGB colour conversion.
      5. Normalise to [0, 1].
      6. Apply ImageNet mean/std normalisation.
      7. Reshape from HWC → CHW and add a batch dimension.

    Returns:
        float32 numpy array of shape (1, 3, 224, 224) ready for ONNX inference.
    """
    # 1. Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image — ensure the file is a valid JPEG, PNG, or BMP.")

    # 2. Hair removal
    clean_img = dull_razor_hair_removal(img)

    # 3. Resize
    resized = cv2.resize(clean_img, target_size)

    # 4. BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 5. Normalise to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # 6. ImageNet mean/std normalisation (must match training transforms)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    # 7. HWC → CHW, then add batch dimension
    chw_img = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw_img, axis=0)

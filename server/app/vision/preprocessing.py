import cv2
import numpy as np

def dull_razor_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    Implements the Dull Razor algorithm to remove hair from dermatoscopic images.
    
    1. Gray-scale conversion.
    2. Black-Hat morphological filtering (to extract hair fibers).
    3. Hair mask creation & thresholding.
    4. Telea image inpainting to restore the skin texture underneath.
    
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
    
    # 3. Hair mask creation & thresholding
    # Enhance the extracted hairs
    # We apply a Gaussian blur to reduce noise, then threshold
    blurred = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, hair_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    
    # Optional: Dilate the mask slightly to ensure edges of hair are covered
    kernel_dilate = np.ones((3,3), np.uint8)
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    
    # 4. Image inpainting (Telea method)
    # Restore the skin texture based on neighboring pixels
    restored_image = cv2.inpaint(image, hair_mask, 6, cv2.INPAINT_TELEA)
    
    return restored_image

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline for incoming medical images.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image data")
        
    # Apply Dull Razor
    clean_img = dull_razor_hair_removal(img)
    
    # Resize to target input size
    resized = cv2.resize(clean_img, target_size)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] range
    normalized = rgb.astype(np.float32) / 255.0
    
    # Apply ImageNet mean and std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Change format from HWC to CHW for PyTorch/ONNX
    chw_img = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batched_img = np.expand_dims(chw_img, axis=0)
    
    return batched_img

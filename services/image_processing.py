import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def detect_blurriness(image_np):
    """Check if the image is blurry using the Laplacian variance method."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.debug(f"Laplacian Variance (Blurriness Score): {laplacian_var}")
    threshold = 100
    if laplacian_var < threshold:
        return False, "Image is blurry"
    return True, "Image is clear"


def detect_contrast(image_np):
    """Analyze contrast using the 5th-95th percentile spread."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray.size
    cum_hist = np.cumsum(hist)
    lower_idx = np.searchsorted(cum_hist, 0.05 * total_pixels)
    upper_idx = np.searchsorted(cum_hist, 0.95 * total_pixels)
    contrast_ratio = (upper_idx - lower_idx) / 255.0
    logger.debug(f"Contrast Ratio (5th-95th percentiles): {contrast_ratio:.4f}")
    lower_threshold = 0.15
    upper_threshold = 0.85
    if contrast_ratio < lower_threshold:
        return False, "Low contrast detected"
    elif contrast_ratio > upper_threshold:
        return False, "High contrast detected"
    return True, "Contrast is acceptable"


def detect_shadows(image_np):
    """Detect shadows in the background or face by analyzing dark regions."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    shadow_threshold = 50
    shadow_pixels = np.sum(gray < shadow_threshold)
    shadow_ratio = shadow_pixels / (gray.shape[0] * gray.shape[1])
    logger.debug(f"Shadow Ratio: {shadow_ratio:.4f}")
    if shadow_ratio > 0.1:
        return False, "Shadows detected on the face or background"
    return True, "No significant shadows detected"


def process_face_segmentation(image_np):
    """Performs person segmentation and validates background color."""
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        logger.debug("Segmentation results obtained.")
        mask = results.segmentation_mask
        condition = mask > 0.5
        mask_area = np.sum(condition)
        total_area = mask.shape[0] * mask.shape[1]
        mask_percentage = mask_area / total_area
        logger.debug("Mask percentage: %.2f", mask_percentage)

        # Segment the person from the background
        bg_removed = np.zeros_like(image_np)
        segmented_image = np.where(condition[..., None], image_np, bg_removed)

        # Check if the background is white or near white
        bg_mask = ~condition
        background_pixels = image_np[bg_mask]
        total_bg_pixels = background_pixels.shape[0]
        valid_white_pixels = np.sum((background_pixels >= 245) & (background_pixels <= 255))
        threshold = 0.9
        is_background_white = (valid_white_pixels / total_bg_pixels) >= threshold if total_bg_pixels > 0 else False
        logger.debug("Background validation: %s", is_background_white)

    return is_background_white, mask_percentage, segmented_image

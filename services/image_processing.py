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
        return False, "image_is_blurry"
    return True, "image_is_clear"


def detect_overexposure(image_np, brightness_threshold=245, overexposed_ratio=0.6):
    """Detects overexposure by checking how many pixels are near maximum brightness."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Count pixels that are nearly white
    bright_pixels = np.sum(gray > brightness_threshold)
    total_pixels = gray.size

    bright_ratio = bright_pixels / total_pixels

    if bright_ratio > overexposed_ratio:
        return False, "overexposed_detected"

    return True, "exposure_is_acceptable"


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
    """Performs person segmentation, validates background color, and checks head margin at the top."""
    height, width, _ = image_np.shape

    # Perform segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        mask = results.segmentation_mask
        condition = mask > 0.5
        mask_area = np.sum(condition)
        total_area = mask.shape[0] * mask.shape[1]
        mask_percentage = mask_area / total_area

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


    # Ensure space above the head
    y_indices, _ = np.where(condition)
    head_has_top_margin = False
    if y_indices.size > 0:
        min_y = np.min(y_indices)
        head_has_top_margin = min_y > 1


    return is_background_white, mask_percentage, segmented_image, head_has_top_margin

import cv2
import io
import binascii
from PIL import Image
import mediapipe as mp
import logging

from services.image_processing import (
    detect_blurriness,
    detect_contrast,
    detect_shadows,
    process_face_segmentation
)
from services.face_validation import validate_head_orientation_and_expression

logger = logging.getLogger(__name__)


def process_image_validation(image_np):
    # --- Face Detection using MediaPipe FaceDetection ---
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.75) as face_detection:
        face_results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        logger.debug("Face detection results: %s", face_results)
        if not face_results.detections:
            raise Exception("No face detected in the image")
        elif len(face_results.detections) > 1:
            raise Exception("Multiple faces detected in the image")

    # --- Person Segmentation & Background Check ---
    is_background_white, mask_percentage, segmented_image = process_face_segmentation(image_np)

    # Enforce mask coverage between 60% and 85% (approx. 70%-80%)
    if not (0.50 <= mask_percentage <= 0.85):
        raise Exception("Face does not cover between 70% to 80% of the image")

    # --- Image Quality Validations ---
    is_clear, blur_message = detect_blurriness(image_np)
    is_good_contrast, contrast_message = detect_contrast(image_np)
    is_no_shadows, shadow_message = detect_shadows(image_np)

    # --- Head Orientation and Expression Validation ---
    is_valid, head_message, face_mesh_hex = validate_head_orientation_and_expression(image_np)

    if is_valid:
        logger.debug("Head orientation and expression validation successful: %s", head_message)
    else:
        logger.debug("Head orientation and expression validation failed: %s", head_message)

    # --- Convert segmented image to hex string ---
    segmented_pil = Image.fromarray(segmented_image)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="JPEG")
    buf.seek(0)
    hex_image = binascii.hexlify(buf.read()).decode('utf-8')

    result = {
        "message": "Image processed successfully",
        "is_background_accepted": bool(is_background_white),
        "mask_percentage": mask_percentage,
        "head_validation": head_message,
        "blur_status": blur_message,
        "contrast_status": contrast_message,
        "shadow_status": shadow_message,
        "hex_image": hex_image,
        "face_mesh_hex": face_mesh_hex,
    }
    logger.debug("Validation result: %s", result)
    return result

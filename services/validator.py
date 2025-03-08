import cv2
import io
import binascii
from PIL import Image
import mediapipe as mp
import logging

from scipy.spatial.distance import is_valid_y

from services.image_processing import (
    detect_blurriness,
    detect_overexposure,
    detect_shadows,
    process_face_segmentation
)
from services.face_validation import validate_head_orientation_and_expression

logger = logging.getLogger(__name__)


def process_image_validation(image_np):
    # --- Face Detection using MediaPipe FaceDetection ---
    mp_face_detection = mp.solutions.face_detection
    validation_mssg = ""
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.75) as face_detection:
        face_results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        logger.debug("Face detection results: %s", face_results)
        is_valid_0 = True
        if not face_results.detections:
            is_valid_0 = False
            validation_mssg = "no_face_in_image"
        elif len(face_results.detections) > 1:
            is_valid_0 = False
            validation_mssg = "multiple_faces_in_image"

    # --- Person Segmentation & Background Check ---
    is_background_white, mask_percentage, segmented_image, head_has_margin = process_face_segmentation(image_np)

    is_valid_5 = False
    if 0.50 <= mask_percentage <= 0.85:
        is_valid_5 = True

    # --- Image Quality Validations ---
    is_valid_4, blur_message = detect_blurriness(image_np)
    is_valid_3, contrast_message = detect_overexposure(image_np)
    # is_no_shadows, shadow_message = detect_shadows(image_np)

    # --- Head Orientation and Expression Validation ---
    is_valid_2, head_message, face_mesh_hex = validate_head_orientation_and_expression(image_np)

    # --- Convert segmented image to hex string ---
    segmented_pil = Image.fromarray(segmented_image)
    buf = io.BytesIO()
    segmented_pil.save(buf, format="JPEG")
    buf.seek(0)
    hex_image = binascii.hexlify(buf.read()).decode('utf-8')

    is_valid = False

    if (is_valid_0 and head_has_margin and is_background_white and is_valid_2 and is_valid_3 and is_valid_5
            and validation_mssg != "multiple_faces_in_image"):
        is_valid = True

    result = {
        "message": "Image processed successfully",
        "is_valid" : is_valid,
        "is_background_accepted": bool(is_background_white),
        "head_validation": head_message,
        "blur_status": blur_message,
        "contrast_status": contrast_message,
        "head_margin": bool(head_has_margin),
        "hex_image": hex_image,
        "face_mesh_hex": face_mesh_hex,
    }
    logger.debug("Validation result: %s", result)
    return is_valid, result

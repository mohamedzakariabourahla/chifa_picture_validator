import cv2
import numpy as np
import mediapipe as mp
import binascii
import logging

logger = logging.getLogger(__name__)
mp_face_mesh = mp.solutions.face_mesh

def validate_head_orientation_and_expression(image_np):
    logger.debug("Starting head orientation and expression validation.")
    try:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.75) as face_mesh:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return False, "No face landmarks detected", None

            landmarks = results.multi_face_landmarks[0]
            # Calculate head tilt using eye landmarks
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[133]
            angle = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)

            # Calculate mouth openness using lip landmarks
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            mouth_open_distance = abs(upper_lip.y - lower_lip.y)

            # Annotate face with landmarks
            annotated_image = image_np.copy()
            try:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                        mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                    )
            except Exception as e:
                logger.error("Error during landmark drawing: %s", e)
                return False, f"Error drawing face mesh: {e}", None

            # Convert the annotated image to a hex string
            _, buffer = cv2.imencode('.jpg', annotated_image)
            hex_image = binascii.hexlify(buffer).decode('utf-8')

            # Validate head tilt and mouth openness
            if abs(angle) > 0.1:
                return False, "Head is tilted", hex_image
            if mouth_open_distance > 0.05:
                return False, "Mouth is open", hex_image

            return True, "Face is valid", hex_image
    except Exception as e:
        logger.error("Error during head orientation validation: %s", str(e), exc_info=True)
        return False, f"Validation error: {str(e)}", None

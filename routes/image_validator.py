from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import logging
from services.validator import process_image_validation

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/")
async def process_image(file: UploadFile = File(...)):
    logger.debug("Received file upload request.")
    try:
        if not file.content_type.startswith("image/"):
            logger.debug("Invalid file type: %s", file.content_type)
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        logger.debug("Image successfully loaded and converted to numpy array.")

        result = process_image_validation(image_np)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

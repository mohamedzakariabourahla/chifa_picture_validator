from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import zipfile
import os
import shutil
import logging
from services.validator import process_image_validation


logger = logging.getLogger(__name__)
router = APIRouter()

VALID_DIR = "valid"
INVALID_DIR = "invalid"

# Ensure directories exist
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(INVALID_DIR, exist_ok=True)

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

        is_valid, result = process_image_validation(image_np)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)




@router.post("/zip")
async def process_zip_file(file: UploadFile = File(...)):
    logger.debug("Received ZIP file upload request.")
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a ZIP archive")

    results = []
    try:
        temp_dir = "temp_extracted"
        os.makedirs(temp_dir, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(await file.read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for img_name in os.listdir(temp_dir):
            img_path = os.path.join(temp_dir, img_name)
            if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            with Image.open(img_path) as image:
                image = image.convert("RGB")
                image_np = np.array(image)
                is_valid, result = process_image_validation(image_np)
                save_path = os.path.join(VALID_DIR if is_valid else INVALID_DIR, img_name)

                # Save the image to the corresponding directory
                image.save(save_path)

                results.append({"filename": img_name, "valid": is_valid})

        shutil.rmtree(temp_dir)  # Cleanup extracted files
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error("Unhandled exception: %s", str(e), exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
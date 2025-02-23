from fastapi import FastAPI
from routes.image_validator import router as image_validator_router

app = FastAPI()
app.include_router(image_validator_router, prefix="/chifaPictureValidator")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7070)

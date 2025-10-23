import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.routes.prediction import router as prediction_router
from src.routes.province import router as province_router
from src.config import settings
from src.utils.logger import logger

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(province_router)
app.include_router(prediction_router)

for folder in [settings.UPLOAD_FOLDER, settings.GRAD_CAM_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Ensured folder exists: {folder}")

# Mount static files
app.mount("/static/uploads", StaticFiles(directory=settings.UPLOAD_FOLDER), name="uploads")
app.mount("/static/gradcam", StaticFiles(directory=settings.GRAD_CAM_FOLDER), name="gradcam")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

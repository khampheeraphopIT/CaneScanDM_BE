import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.prediction import router as prediction_router
from src.routes.province import router as province_router
from src.utils.logger import logger
from src.config import settings
from fastapi.responses import FileResponse
from fastapi import HTTPException

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Routes
app.include_router(province_router)
app.include_router(prediction_router)

# ✅ Static file serving (uploads, gradcam, etc.)
@app.get("/{path:path}")
async def serve_file(path: str):
    file_path = os.path.join(os.getcwd(), path)
    if os.path.exists(file_path):
        logger.info(f"Serving file: {file_path}")
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

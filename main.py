from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.routes import predict, diseases
from src.services.rate_tracker import rate_tracker

app = FastAPI(
    title="CaneScan DM API",
    description="ระบบตรวจโรคใบอ้อยด้วย AI Vision",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(diseases.router, prefix="/api", tags=["Diseases"])


@app.get("/")
async def root():
    return {
        "message": "CaneScan DM API v2.0",
        "docs": "/docs"
    }


@app.get("/api/rate-limit")
async def get_rate_limit():
    """Get current rate limit status"""
    return rate_tracker.to_dict()

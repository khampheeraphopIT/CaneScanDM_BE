from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.config.database import get_db, engine
from src.models.base import Base
from src.models.user import User
from src.models.prediction import PredictionHistory
from src.routes import predict, diseases, auth
from src.controllers.prediction_controller import PredictionController
from src.controllers.rate_limit_controller import RateLimitController


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on startup"""
    print("⏳ Checking database tables...")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables ready!")
    except Exception as e:
        # 🔥 พิมพ์ Error ออกมาดูว่าติดอะไร (เช่น SSL, Password, หรือ Connection)
        print(f"❌ Database initialization failed: {str(e)}")
        # ❗ ไม่ raise → ให้ app ยัง start ได้
        # ถ้า raise → Render จะ kill service ทันที

    yield


app = FastAPI(
    title="CaneScan DM API",
    description="ระบบตรวจโรคใบอ้อยด้วย AI Vision",
    version="2.0.0",
    lifespan=lifespan
)

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Routes
# --------------------------------------------------
app.include_router(auth.router, tags=["Authentication"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(diseases.router, tags=["Diseases"])


@app.get("/")
async def root():
    return {
        "message": "CaneScan DM API v2.0",
        "docs": "/docs"
    }


@app.get("/rate-limit")
async def get_rate_limit():
    """Get current rate limit status"""
    return await RateLimitController.get_status()


@app.get("/history")
async def get_history(db: AsyncSession = Depends(get_db)):
    """Fetch prediction history from database"""
    return await PredictionController.get_history(db)

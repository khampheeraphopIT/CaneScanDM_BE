from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from src.config.database import get_db
from src.controllers.prediction_controller import PredictionController
from src.routes.auth import get_current_user

router = APIRouter()
security = HTTPBearer(auto_error=False)


@router.post("/predict")
async def predict_disease(
    image: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Route to handle disease prediction
    """
    # Get current user if authenticated (optional)
    user = await get_current_user(credentials, db) if credentials else None
    user_id = user.id if user else None
    
    return await PredictionController.predict_disease(image, db, user_id=user_id)


import os
import json
from fastapi import APIRouter, HTTPException
from src.utils.logger import logger

router = APIRouter(prefix="/provinces", tags=["Provinces"])

try:
    json_path = os.path.join(os.getcwd(), "api_province.json")
    with open(json_path, "r", encoding="utf-8") as f:
        provinces_data = json.load(f)
    provinces = sorted([prov["name_th"] for prov in provinces_data])
except Exception as e:
    logger.error(f"Failed to load api_province.json: {e}")
    raise HTTPException(status_code=500, detail="Failed to load province data")

@router.get("")
async def get_provinces():
    return provinces
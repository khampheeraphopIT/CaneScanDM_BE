from fastapi import APIRouter
from src.controllers.disease_controller import DiseaseController

router = APIRouter()

@router.get("/diseases")
async def get_diseases():
    """
    Fetch all sugarcane disease information
    """
    diseases = await DiseaseController.get_all_diseases()
    return {
        "success": True,
        "diseases": diseases
    }

@router.get("/diseases/{disease_key}")
async def get_disease(disease_key: str):
    """
    Fetch specific disease information by key
    """
    disease = await DiseaseController.get_disease_by_key(disease_key)
    if not disease:
        return {
            "success": False,
            "error": "ไม่พบข้อมูลโรค"
        }
    return {
        "success": True,
        "disease": disease
    }

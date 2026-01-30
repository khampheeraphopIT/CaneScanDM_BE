from src.models.disease_data import DISEASES

class DiseaseController:
    @staticmethod
    async def get_all_diseases():
        """
        Get all disease definitions
        """
        return DISEASES

    @staticmethod
    async def get_disease_by_key(disease_key: str):
        """
        Fetch a specific disease by key
        """
        key = disease_key.lower().replace(" ", "_")
        return DISEASES.get(key)

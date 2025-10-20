import os

class Settings:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    GRAD_CAM_FOLDER = os.path.join(os.getcwd(), "gradcam")
    MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "model", "model_20251019_122736.pth")
    CSV_LOG_PATH = os.path.join(os.getcwd(), "prediction_logs.csv")
    ALLOW_ORIGINS = [
        "http://localhost:5173",       
        "http://10.0.2.2:8000",        
        "http://*.*.*.*:8000",         
        "https://your-app-domain.com", 
    ]
    HOST = "0.0.0.0"
    PORT = 8000

settings = Settings()

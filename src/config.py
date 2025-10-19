import os

class Settings:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    GRAD_CAM_FOLDER = os.path.join(os.getcwd(), "gradcam")
    MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "best_val_model.pth")
    CSV_LOG_PATH = os.path.join(os.getcwd(), "prediction_logs.csv")
    ALLOW_ORIGINS = ["http://localhost:5173"]
    HOST = "0.0.0.0"
    PORT = 8000

settings = Settings()

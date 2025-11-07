# CaneScanDM_BE

# Run Project
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Deploy Command
web: uvicorn main:app --host 0.0.0.0 --port $PORT


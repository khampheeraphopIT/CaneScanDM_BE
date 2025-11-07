# Use a slim Python base
FROM python:3.13.9-slim

# Run Python in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyTorch / grad-cam
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (pip will pull pre-built wheels)
RUN pip install --no-cache-dir -r requirements.txt

# Copy only your app code (exclude .venv, data, notebooks)
COPY . .

# Expose port (optional, depends on your app)
EXPOSE 8000

# Start Gunicorn web server
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "1"]

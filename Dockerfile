FROM python:3.13.9
WORKDIR /app

# Copy only requirements.txt first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

CMD ["gunicorn", "main:app"]

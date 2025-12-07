# --- BASE IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install all Python dependencies including uvicorn and fastapi in base environment
RUN pip install --no-cache-dir -r requirements.txt uvicorn fastapi python-dotenv requests pandas

# Copy application code
COPY . .

# Expose port (Railway uses $PORT)
EXPOSE 8000

# --- FASTAPI CMD ---
# Adjust path depending on where your main.py is:
# If main.py is at root folder: uvicorn main:app
# If main.py is inside app/ folder: uvicorn app.main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

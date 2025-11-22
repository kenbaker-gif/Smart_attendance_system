# Base image
FROM python:3.10-slim

# Install system dependencies for dlib/face_recognition
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file (for caching purposes)
COPY requirements.txt /app/

# --- START OF OPTIMIZED INSTALLATION ---

# 1. Install the heavy, slow-to-compile packages (dlib, face-recognition)
# This layer will be slow the FIRST time (the 916.1s part), but will be cached for subsequent builds
# UNLESS dlib's package version is changed.
RUN pip install --no-cache-dir dlib face-recognition-models face-recognition

# 2. Install the remaining dependencies from requirements.txt
# This will be very fast, and is only re-run if requirements.txt changes.
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

# --- END OF OPTIMIZED INSTALLATION ---

# Copy project source code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
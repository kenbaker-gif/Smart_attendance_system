FROM python:3.10-slim

# Install system dependencies needed by face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# === Install precompiled Dlib wheel (FAST, no compilation) ===
RUN pip install --no-cache-dir https://github.com/datamllab/rlcard/releases/download/v1.0.4/dlib-19.24.0-cp310-cp310-manylinux_2_17_x86_64.whl

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
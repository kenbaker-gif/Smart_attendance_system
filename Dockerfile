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
RUN pip install --no-cache-dir https://files.pythonhosted.org/packages/41/ed/9cb28f8e3af1c477a7e2e6d2f629d153576f7462c3df1da7efd7d446257f/dlib-19.24.0-cp310-cp310-manylinux_2_31_x86_64.whl

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
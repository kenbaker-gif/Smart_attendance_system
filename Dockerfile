# --- STAGE 1: BUILDER (Heavy Lifting) ---
FROM continuumio/miniconda3:latest AS builder

# Install minimal system dependencies for building dlib/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (excluding dlib and opencv)
COPY requirements.txt .

# Install heavy dependencies using conda
RUN conda install -c conda-forge dlib=19.24 opencv -y --no-update-deps --quiet

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Optional: clean conda caches to reduce size
RUN conda clean -afy

# --- STAGE 2: FINAL (Minimal Runtime) ---
FROM python:3.10-slim

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the prebuilt conda environment from builder
COPY --from=builder /opt/conda /opt/conda

# Ensure Conda binaries are in PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

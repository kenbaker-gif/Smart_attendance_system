# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements (excluding dlib/opencv)
COPY requirements.txt .

# Install heavy dependencies using conda
RUN conda install -c conda-forge dlib=19.24 opencv -y --no-update-deps --quiet

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Clean conda cache to reduce size
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

# Copy prebuilt conda environment
COPY --from=builder /opt/conda /opt/conda

ENV PATH="/opt/conda/bin:$PATH"

# Copy app code
COPY . .

# Use dynamic PORT for Railway
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Start app using correct module path
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
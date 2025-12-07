# --- STAGE 1: BUILDER (Install dependencies) ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create conda environment and install dependencies
RUN conda create -n app_env python=3.11 -y && \
    conda install -n app_env -c conda-forge dlib=19.24 opencv -y --quiet && \
    conda install -n app_env -c conda-forge uvicorn fastapi python-dotenv requests pandas -y && \
    /opt/conda/envs/app_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL IMAGE (Runtime only) ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code
COPY . .

# Expose port (Railway provides $PORT)
EXPOSE 8000

# --- RUN FASTAPI BACKEND ---
# Use conda run to ensure uvicorn is in the environment
CMD sh -c "conda run -n app_env uvicorn main:app --host 0.0.0.0 --port $PORT"

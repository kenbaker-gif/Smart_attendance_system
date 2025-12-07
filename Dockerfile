# --- STAGE 1: BUILDER (Install dependencies) ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create environment with Python 3.11 and install all dependencies including uvicorn/fastapi
RUN conda create -n app_env python=3.11 -y && \
    conda install -n app_env -c conda-forge dlib=19.24 opencv -y --quiet && \
    conda install -n app_env -c conda-forge python-dotenv requests pandas -y && \
    /opt/conda/envs/app_env/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/conda/envs/app_env/bin/pip install --no-cache-dir uvicorn fastapi && \
    conda clean -afy

# --- STAGE 2: FINAL IMAGE (Runtime only) ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code
COPY . .

# Expose port (Railway uses $PORT)
EXPOSE 8000

# --- RUN FASTAPI BACKEND ---
# Use uvicorn from base PATH (already installed in app_env)
# Adjust main path depending on your file structure
# If main.py is at root:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# If main.py is inside app/ folder, use:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

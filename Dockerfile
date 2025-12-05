# --- STAGE 1: BUILDER (Install dependencies) ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install minimal system dependencies needed for dlib/face-recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create a minimal environment with only necessary packages
RUN conda create -n app_env python=3.11 -y && \
    conda install -n app_env -c conda-forge dlib=19.24 opencv -y --quiet && \
    /opt/conda/envs/app_env/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# --- STAGE 2: FINAL IMAGE (Runtime only) ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy the environment from builder
COPY --from=builder /opt/conda /opt/conda

# Update PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code last
COPY . .

# Streamlit settings
ENV PORT=8501
EXPOSE $PORT

CMD ["sh", "-c", "streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$PORT"]

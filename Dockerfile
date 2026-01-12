# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxrender1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Update Conda and set channel priority
RUN conda update -n base -c defaults conda -y && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Install TensorFlow and OpenCV from conda-forge
# Note: DeepFace often requires specific TF versions; 2.13 is a solid choice.
RUN conda install -y tensorflow=2.13.0 opencv && conda clean -afy

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt deepface

# --- STAGE 2: FINAL IMAGE ---
# Use a slimmer base if possible, but keeping miniconda for path consistency
FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire conda environment from builder
COPY --from=builder /opt/conda /opt/conda

# Ensure path is set
ENV PATH="/opt/conda/bin:$PATH"

# Copy application code
COPY . .

# Streamlit config
ENV PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
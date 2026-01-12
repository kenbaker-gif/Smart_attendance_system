# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install build tools and GL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and OpenCV via Conda
RUN conda install -y python=3.10 opencv -c conda-forge && conda clean -afy

# Install ML libraries via Pip
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    deepface \
    tf-keras \
    streamlit \
    supabase

# --- BAKE MODELS HERE ---
# This Python command forces DeepFace to download the weights during build.
# We download VGG-Face and RetinaFace (the most common defaults).
RUN python -c "from deepface import DeepFace; \
    DeepFace.build_model('VGG-Face'); \
    DeepFace.build_model('RetinaFace')"

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install runtime GL dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment and the downloaded weights
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# Set environment paths
ENV PATH="/opt/conda/bin:$PATH"
ENV HOME="/root"

# Copy application code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
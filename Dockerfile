# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install system dependencies including the updated C++ standard library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and OpenCV from conda-forge
# This ensures we have a modern build of OpenCV
RUN conda install -y python=3.10 opencv -c conda-forge && conda clean -afy

# Install ML libraries via Pip
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    deepface \
    tf-keras \
    streamlit \
    supabase

# Bake Models: Download weights into /root/.deepface during build
ENV TF_CPP_MIN_LOG_LEVEL=3
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face'); DeepFace.build_model('RetinaFace')"

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Sync system libraries with Stage 1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment and pre-trained weights
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# Set environment paths
# We point LD_LIBRARY_PATH to conda's lib to prevent CXXABI mismatches
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV HOME="/root"
ENV TF_CPP_MIN_LOG_LEVEL=3

# Copy application code
COPY . .

# Streamlit Port
ENV PORT=8501
EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
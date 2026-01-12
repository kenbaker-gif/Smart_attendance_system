# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# 1. Install system-level dependencies for OpenCV and C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3.10 and the modern C++ standard library (Fixes CXXABI error)
RUN conda install -y -c conda-forge \
    python=3.10 \
    opencv \
    libstdcxx-ng=13.2.0 \
    && conda clean -afy

# 3. Install Pip dependencies required by your app.py
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    deepface \
    tf-keras \
    streamlit \
    supabase \
    python-dotenv

# 4. BAKE MODELS: Pre-cache the specific weights used in app.py
# We use 'modeling' directly to ensure retinaface builds correctly.
RUN export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH && \
    python -c "from deepface import DeepFace; \
    from deepface.modules import modeling; \
    DeepFace.build_model('Facenet512'); \
    modeling.build_model(task='face_detector', model_name='retinaface')"

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install runtime GL and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the optimized Conda environment and pre-downloaded weights
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# Set environment variables for runtime stability
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV HOME="/root"
ENV DEEPFACE_HOME="/root"
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Create persistent data and log directories
RUN mkdir -p data/raw_faces logs

# Copy your application source code
COPY . .

# Streamlit configuration
ENV PORT=8501
EXPOSE 8501

# Command to launch the application
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
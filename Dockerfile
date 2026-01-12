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

# 1. Install runtime system dependencies (Crucial for OpenCV on Cloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the optimized Conda environment and pre-downloaded weights
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# 3. Set environment variables for runtime stability
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV HOME="/root"
ENV DEEPFACE_HOME="/root"
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# 4. Create persistent data and log directories
RUN mkdir -p data/raw_faces logs

# 5. Copy your application source code
COPY . .

# 6. RAILWAY PORT FIX: 
# Using the shell form of CMD allows the $PORT environment variable 
# provided by Railway to be injected into the Streamlit command.
CMD streamlit run streamlit/app.py --server.port=$PORT --server.address=0.0.0.0
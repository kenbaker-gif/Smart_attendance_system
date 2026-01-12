# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3.10, OpenCV, and the updated C++ lib
RUN conda install -y -c conda-forge \
    python=3.10 \
    opencv \
    libstdcxx-ng=13.2.0 \
    && conda clean -afy

# 3. Install Pip dependencies
RUN pip install --no-cache-dir tensorflow==2.13.0 deepface tf-keras streamlit supabase python-dotenv

# 4. THE FIX: Force the library path specifically for the model bake command
RUN export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH && \
    python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); DeepFace.build_model('RetinaFace')"

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Core runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy everything from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# Set Global Environment Paths (Crucial for the app to run)
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV DEEPFACE_HOME="/root"
ENV PYTHONUNBUFFERED=1

# Create data directories
RUN mkdir -p data/raw_faces logs

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
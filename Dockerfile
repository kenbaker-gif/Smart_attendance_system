# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and the Modern C++ library (Fixes CXXABI error)
RUN conda install -y -c conda-forge \
    python=3.10 \
    opencv \
    libstdcxx-ng=13.2.0 \
    && conda clean -afy

# Install Pip dependencies from your script
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    deepface \
    tf-keras \
    streamlit \
    supabase \
    python-dotenv

# Bake the EXACT models your app.py uses
ENV TF_CPP_MIN_LOG_LEVEL=3
RUN python -c "from deepface import DeepFace; \
    DeepFace.build_model('Facenet512'); \
    DeepFace.build_model('RetinaFace')"

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Ensure runtime system libs are present
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the environment and pre-trained weights
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.deepface /root/.deepface

# Set Environment Variables
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV HOME="/root"
ENV DEEPFACE_HOME="/root"
ENV TF_CPP_MIN_LOG_LEVEL=3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw_faces logs

EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
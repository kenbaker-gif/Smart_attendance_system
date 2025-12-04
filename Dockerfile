# --- STAGE 1: BUILDER ---
FROM continuumio/miniconda3:latest AS builder

WORKDIR /app

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gfortran \
    libgl1 libglib2.0-0 libxext6 libx11-6 \
    libsm6 libxrender1 libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install heavy ML dependencies via conda first
RUN conda install -y tensorflow opencv && conda clean -afy

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: FINAL IMAGE ---
FROM continuumio/miniconda3:latest

WORKDIR /app

# Reinstall minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxext6 libx11-6 \
    libsm6 libxrender1 libopenblas0 liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy project code
COPY . .

# Expose Streamlit and FastAPI ports
ENV STREAMLIT_PORT=8501
ENV FASTAPI_PORT=8000
EXPOSE $STREAMLIT_PORT $FASTAPI_PORT

# Start both FastAPI and Streamlit using & to run in background
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port $FASTAPI_PORT & streamlit run streamlit/app.py --server.address=0.0.0.0 --server.port=$STREAMLIT_PORT"

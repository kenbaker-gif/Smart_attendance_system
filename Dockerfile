FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libopenblas-dev liblapack-dev wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose ports
ENV STREAMLIT_PORT=8501
ENV FASTAPI_PORT=$PORT
EXPOSE $STREAMLIT_PORT
EXPOSE $FASTAPI_PORT

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "streamlit run streamlit/app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 & uvicorn app.main:app --host 0.0.0.0 --port $FASTAPI_PORT"]

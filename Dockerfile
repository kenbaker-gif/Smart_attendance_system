FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose a port for Railway; the actual port will be dynamic
EXPOSE 8501

# Run Streamlit on Railway's dynamic PORT
CMD ["sh", "-c", "streamlit run admin-panel.py --server.address=0.0.0.0 --server.port=${PORT}"]
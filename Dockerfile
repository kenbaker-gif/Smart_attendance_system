# -----------------------------
# Base image: Python 3.11 slim
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Set working directory inside container
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies for Streamlit and common Python packages
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy the app code into the container
# -----------------------------
COPY . .

# -----------------------------
# Expose a port for Railway; the actual port will be dynamic
# -----------------------------
EXPOSE 8501  # fallback for local testing only

# -----------------------------
# Run Streamlit on Railway's dynamic PORT
# -----------------------------
# Railway injects PORT environment variable automatically in production
# The app will listen on 0.0.0.0 so it's reachable externally
# The ${PORT} syntax ensures your app works on both Railway and locally
# -----------------------------
CMD ["sh", "-c", "streamlit run admin-panel.py --server.address=0.0.0.0 --server.port=${PORT}"]
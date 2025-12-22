# --- BASE IMAGE ---
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies for pandas/supabase
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Railway uses a dynamic port
EXPOSE 8501

# --- STREAMLIT RUN COMMAND ---
# Point this exactly to your new admin file
CMD ["sh", "-c", "streamlit run admin-panel.py --server.port $PORT --server.address 0.0.0.0"]
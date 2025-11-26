FROM continuumio/miniconda3

WORKDIR /app

# Install dlib + dependencies using conda (precompiled)
RUN conda install -c conda-forge dlib opencv -y

# Copy requirements without dlib
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
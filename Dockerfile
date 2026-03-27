FROM python:3.10-slim

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY deploy.prototxt .
COPY res10_300x300_ssd_iter_140000.caffemodel .

# Copy application code
COPY app.py .
COPY helpers.py .

# Copy frontend
COPY static/ ./static/

# Create dataset and embeddings dirs
RUN mkdir -p dataset

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

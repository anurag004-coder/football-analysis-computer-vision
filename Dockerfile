# Multi-Stage Dockerfile for Football Analytics CV & Dashboard
FROM python:3.11-slim

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and assets
COPY src/ ./src/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY assets/ ./assets/
COPY analyze.py pipeline.py serve_dashboard.py dashboard.html matches.json ./

# Expose HTTP port for the web dashboard & REST API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/matches || exit 1

# Start the dashboard server
CMD ["python", "serve_dashboard.py", "--port", "8000"]

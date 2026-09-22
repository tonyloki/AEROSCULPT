# AeroSculpt Production Dockerfile
FROM python:3.11-slim

# Install system dependencies: GDAL, PROJ, OpenCV dependencies, FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

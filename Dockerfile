# Use official lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workspace working directory
WORKDIR /app

# Install system dependencies required by OpenCV and graphics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, templates, assets, and trained models
COPY . /app/

# Expose default application port
EXPOSE 8000

# Health check to ensure FastAPI application server status
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch Uvicorn web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

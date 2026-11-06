# ==============================================================================
# Stage 1: Build React Frontend (Vite)
# ==============================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies using package lockfile
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend source code
COPY frontend/ .

# Bake empty VITE_API_URL so frontend uses relative URLs (/api/...) for same-origin
ENV VITE_API_URL=""

# Build React production bundle into /app/frontend/dist
RUN npm run build

# ==============================================================================
# Stage 2: Final Production Image (FastAPI Backend + Nginx + Supervisor)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=10000

WORKDIR /app

# Install system dependencies, OpenCV runtime libraries, Nginx, Supervisor, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default \
    && rm -rf /usr/share/nginx/html/*

# Install Python backend dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application source, configuration, utilities, assets, and models
COPY app.py config.py /app/
COPY utils/ /app/utils/
COPY assets/ /app/assets/
COPY models/ /app/models/

# Copy React production build from Stage 1 into Nginx web root
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Copy Nginx template and Supervisor configurations
COPY frontend.nginx.conf /etc/nginx/templates/frontend.nginx.conf.template
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script and normalize line endings for Linux execution
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Render injects $PORT (default 10000). Nginx listens on $PORT, Uvicorn on 127.0.0.1:8000
EXPOSE 10000

# Docker healthcheck queries the public Nginx port for /api/health (reverse-proxied to Uvicorn)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-10000}/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

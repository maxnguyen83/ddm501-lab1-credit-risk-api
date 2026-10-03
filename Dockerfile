# =============================================================================
# Credit Default Risk Scoring API
# DDM501 - Lab 1: First ML Product
#
# Build:  docker build -t credit-risk-api .
# Run:    docker compose up --build      (mounts ./models read-only)
# =============================================================================
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# 1. Environment
# -----------------------------------------------------------------------------
#   PYTHONDONTWRITEBYTECODE  no .pyc files in the image
#   PYTHONUNBUFFERED         stdout goes straight to `docker logs`; without it a
#                            container that crashes often appears to have logged
#                            nothing
#   PYTHONPATH               so `from app.main import app` resolves from /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# -----------------------------------------------------------------------------
# 2. Working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# 3. Dependencies BEFORE source
# -----------------------------------------------------------------------------
# requirements.txt changes rarely, the code changes on every commit. Copying it
# alone first means this slow layer stays cached when only app/ changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# 4. Application code
# -----------------------------------------------------------------------------
# models/ is NOT copied: the artifact is mounted at run time (docker-compose.yml),
# so a retrained model ships without rebuilding the image.
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# -----------------------------------------------------------------------------
# 5. Non-root user
# -----------------------------------------------------------------------------
# A container running as root that gets compromised is a host running as root.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

# -----------------------------------------------------------------------------
# 6. Port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# 7. Health check
# -----------------------------------------------------------------------------
# "Process is up" is not enough: without its model the API answers /health with
# 200 but every /predict with 503. So the check passes only when the JSON body
# says model_loaded is true. The slim image has no curl, and Python is already
# here, so the check uses the standard library instead of adding a package.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import json, sys, urllib.request; body = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); sys.exit(0 if body.get('model_loaded') is True else 1)"]

# -----------------------------------------------------------------------------
# 8. Startup command
# -----------------------------------------------------------------------------
# 0.0.0.0, not 127.0.0.1: inside a container, loopback is the container's own,
# so a service bound to it looks healthy in the logs and is unreachable from
# outside. Exec form so uvicorn is PID 1 and receives SIGTERM on `docker stop`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

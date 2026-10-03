# =============================================================================
# Credit Default Risk Scoring API
# DDM501 - Lab 1: First ML Product
#
# TODO: Complete this Dockerfile.
# =============================================================================
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# TODO 1: Set the environment variables
# -----------------------------------------------------------------------------
# Requirements:
#   PYTHONDONTWRITEBYTECODE=1   do not litter the image with .pyc files
#   PYTHONUNBUFFERED=1          flush stdout immediately, so `docker logs`
#                               shows output as it happens instead of when the
#                               buffer fills. Without this, a container that
#                               crashes often appears to have logged nothing.
#   PYTHONPATH=/app             so `from app.main import app` resolves
#

# -----------------------------------------------------------------------------
# TODO 2: Set the working directory
# -----------------------------------------------------------------------------
# Hint: WORKDIR /app

# -----------------------------------------------------------------------------
# TODO 3: Install dependencies
# -----------------------------------------------------------------------------
# Requirements:
#   - copy requirements.txt on its own FIRST, then run pip install,
#     then copy the source
#   - use pip install --no-cache-dir
#

# -----------------------------------------------------------------------------
# TODO 4: Copy the application code
# -----------------------------------------------------------------------------
# Requirements: copy app/, scripts/ and data/
# Note we do NOT copy models/ — see docker-compose.yml for why.

# -----------------------------------------------------------------------------
# TODO 5: Create and switch to a non-root user
# -----------------------------------------------------------------------------
# Requirements:
#   - create a user `appuser` with uid 1000 and a home directory
#   - create /app/models and give appuser ownership of /app
#   - switch to that user with USER
#
# A container running as root that gets compromised is a host running as root.
# This is three lines and it is not optional in production.
#
# -----------------------------------------------------------------------------
# TODO 6: Expose the port
# -----------------------------------------------------------------------------
# Hint: EXPOSE 8000

# -----------------------------------------------------------------------------
# TODO 7: Add a health check
# -----------------------------------------------------------------------------
# Requirements:
#   - check every 30s, time out after 10s, allow 10s of start-up, retry 3 times
#   - it must call GET /health AND check that model_loaded is true
#

# -----------------------------------------------------------------------------
# TODO 8: Set the startup command
# -----------------------------------------------------------------------------
# Requirements: run uvicorn on app.main:app, bound to 0.0.0.0:8000
#
# Binding to 127.0.0.1 inside a container is a classic mistake: the service
# comes up, the logs look perfect, and nothing outside the container can reach
# it. Use 0.0.0.0.
#
# Hint: CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

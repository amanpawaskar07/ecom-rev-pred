FROM python:3.11.18-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/app/src:/app

# Create non-root user (uid 10001)
RUN useradd -u 10001 -m -s /bin/bash appuser

WORKDIR /app

# Install runtime dependencies
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt && rm /tmp/requirements-runtime.txt

# Copy only runtime code and models
COPY src/ /app/src/
COPY api/ /app/api/
COPY ui/ /app/ui/
COPY models/ /app/models/

RUN chown -R appuser:appuser /app

# API target
FROM base AS api
USER 10001
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# UI target
FROM base AS ui
USER 10001
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]


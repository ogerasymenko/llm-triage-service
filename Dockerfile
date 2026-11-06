# Verified: built and run with Docker 29.6.1 / Compose v5.3.0 on aarch64.
# `docker compose up -d` brings up api, mock and redis; /health answers 200 from
# the host. The native path (make dev) is still the documented fallback.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY mock ./mock
COPY samples ./samples

EXPOSE 8080 8081

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

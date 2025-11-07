FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY configs ./configs

COPY pipelines ./pipelines

COPY scripts ./scripts

COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python3", "-m", "scripts.train"]


# FROM python:3.11-slim AS builder

# WORKDIR /app

# RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

# COPY src/ /app/src/
# COPY alembic/ /app/alembic/
# COPY alembic.ini /app/alembic.ini


# # Final stage
# FROM python:3.11-slim

# ENV PYTHONUNBUFFERED=1 \
#     POETRY_VIRTUALENVS_CREATE=false \
#     PIP_NO_CACHE_DIR=1 \
#     PYTHONPATH=/app/src:${PYTHONPATH:-}

# WORKDIR /app

# COPY --from=builder /install /usr/local
# COPY --from=builder /app/src /app/src
# COPY --from=builder /app/alembic /app/alembic
# COPY --from=builder /app/alembic.ini /app/alembic.ini

# COPY init_db.py /app/init_db.py

# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# # Add user and assign ownership
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# USER appuser

# EXPOSE 8000

# ENTRYPOINT ["/entrypoint.sh"]
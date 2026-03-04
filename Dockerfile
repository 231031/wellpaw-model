FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

ENV APP_PORT=50002

CMD gunicorn --bind 0.0.0.0:$APP_PORT --workers 1 --threads 2 app:app
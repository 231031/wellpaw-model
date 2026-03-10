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

RUN mkdir -p /app/model

ENV PORT=50002

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 app:app
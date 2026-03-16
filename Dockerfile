FROM python:3.9-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p /app/model

ENV PORT=50002

CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 4 --timeout 120 app:app
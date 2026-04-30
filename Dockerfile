FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "litellm[proxy]"

COPY config.yaml .

EXPOSE 10000

CMD ["litellm", "--config", "config.yaml", "--port", "10000", "--host", "0.0.0.0"]

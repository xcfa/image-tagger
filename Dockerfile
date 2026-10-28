# Waifu Diffusion image tagger: models from Hugging Face on the first start, kept in /models (mount a volume there).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TAGGER_PORT=8000 TAGGER_MODELS_DIR=/models

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tagger ./tagger

# The models folder belongs to the service's user; a mounted host folder needs to be writable by uid 1000.
RUN useradd --uid 1000 --create-home tagger && mkdir -p /models && chown tagger:tagger /models
USER tagger
VOLUME /models
EXPOSE 8000

# 503 while the models download (a big one takes minutes), 200 once they are loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"TAGGER_PORT\"]}/health', timeout=4)"

CMD ["python", "-m", "tagger"]

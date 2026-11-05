# Builds the index at image build time so the container starts serving immediately.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN python -m finrag.cli download && python -m finrag.cli extract && python -m finrag.cli build \
    && rm -rf data/pdfs data/pages.jsonl
EXPOSE 8080
CMD ["sh", "-c", "uvicorn finrag.api:app --host 0.0.0.0 --port ${PORT:-8080}"]

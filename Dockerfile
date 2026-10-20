FROM node:24-alpine AS widget
WORKDIR /widget
COPY widget/package.json widget/package-lock.json ./
RUN npm ci
COPY widget/ ./
RUN mkdir -p ../app/static && npm run build

FROM node:24-alpine AS site
WORKDIR /build/site
COPY site/package.json site/package-lock.json ./
RUN npm ci
COPY site/ ./
# Tailwind keeps only the classes the landing template uses.
COPY app/templates/landing.html ../app/templates/landing.html
RUN npm run build

FROM python:3.12-slim
WORKDIR /srv
ENV FASTEMBED_CACHE_PATH=/srv/models \
    EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Bake the topic embedding model into the image: topics never leave the server.
RUN python -c "import fastembed, os; fastembed.TextEmbedding(os.environ['EMBEDDING_MODEL'])"
COPY app/ app/
COPY locales/ locales/
COPY alembic.ini ./
COPY migrations/ migrations/
COPY --from=widget /app/static/index.html app/static/index.html
COPY --from=site /build/app/static/site/ app/static/site/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]

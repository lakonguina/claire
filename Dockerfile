FROM node:24-alpine AS widget
WORKDIR /widget
COPY widget/package.json widget/package-lock.json ./
RUN npm ci
COPY widget/ ./
RUN mkdir -p ../app/static && npm run build

FROM python:3.12-slim
WORKDIR /srv
ENV FASTEMBED_CACHE_PATH=/srv/models
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Bake the topic embedding model into the image: topics never leave the server.
COPY app/topics.py app/db.py app/config.py app/vocab.py app/__init__.py app/
RUN python -c "from app import topics; topics._model()"
COPY app/ app/
COPY locales/ locales/
COPY migrations/ migrations/
COPY --from=widget /app/static/index.html app/static/index.html
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--proxy-headers", "--forwarded-allow-ips", "*"]

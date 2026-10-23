FROM node:22-slim AS css
WORKDIR /build
COPY package.json package-lock.json* ./
RUN npm ci
COPY app/static/styles.css ./app/static/styles.css
COPY app/templates ./app/templates
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY --from=css /build/app/static/app.css ./app/static/app.css
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

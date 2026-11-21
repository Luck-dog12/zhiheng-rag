FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY backend ./backend
RUN uv sync --frozen --no-dev
COPY alembic.ini ./
COPY --from=frontend /web/dist ./frontend/dist
RUN useradd --create-home --uid 10001 rag
USER rag
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

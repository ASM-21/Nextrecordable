# nextrecordable API. The image holds code only; the trained database and
# model are mounted from the host at /app/data (see deploy/RUNBOOK.md).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NEXTRECORDABLE_DATA_DIR=/app/data \
    NEXTRECORDABLE_DB_PATH=/app/data/nextrecordable.duckdb \
    NEXTRECORDABLE_MODEL_DIR=/app/data/model \
    NEXTRECORDABLE_ACCOUNTS_DB=/app/data/accounts.sqlite \
    NEXTRECORDABLE_OUTBOX_DIR=/app/data/outbox

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY nextrecordable ./nextrecordable
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
# One process per database file (DuckDB is single-writer).
CMD ["nextrecordable", "serve", "--host", "0.0.0.0", "--port", "8000"]

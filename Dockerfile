# syntax=docker/dockerfile:1

# ---- build stage ------------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime stage: slim, non-root -----------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN groupadd -r app && useradd -r -g app app
COPY --from=builder /app /app
USER app

# The image is a verification/worker job: it runs the adversarial + durability
# eval harness. See docs/architecture.md for the production deployment story
# (LangGraph Platform) — the graph itself is identical either way.
CMD ["governor-agent", "eval"]

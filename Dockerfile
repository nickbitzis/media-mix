# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 mediamix
WORKDIR /app
COPY --from=build --chown=mediamix:mediamix /app /app
# The app keeps everything in memory and writes nothing under /app at runtime.
USER mediamix
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8859 MEDIAMIX_BOOT=200
EXPOSE 8859
# The study (search, calibration, 200 bootstrap draws, budget plans) fits at startup: about 10 to 30 seconds.
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8859/healthz', timeout=2)"
CMD ["mediamix", "serve"]

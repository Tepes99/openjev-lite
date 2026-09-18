FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
# Install deps first (layer-cached): the lock's torch is the CUDA build,
# which also runs on CPU (the VPS has no GPU).
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev
COPY . .
ENV PATH=/app/.venv/bin:$PATH
# The laya model lives on the persistent /data volume, downloaded once on
# first boot by the background pre-warm; it is NOT part of the image.
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

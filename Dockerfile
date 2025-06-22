
# First, build the application in the `/app` directory.
# See `Dockerfile` for details.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install runtime utilities
RUN apt-get update && apt-get install -y \
    nano \
    curl \
    wget \
    less \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
# Copy Python model inference code
COPY model_inference/ /app/
COPY models/anomaly_detection_lstm /app/models
WORKDIR /app

# Default command (change as needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

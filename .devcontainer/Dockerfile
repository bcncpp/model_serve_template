# Stage 1: Base with PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base

# Stage 2: Dev container base (adds tooling)
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04 AS dev

# Copy runtime files from base (with PyTorch + CUDA)
COPY --from=base / / 

# Set arguments and environment
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ENV DEBIAN_FRONTEND=noninteractive
# Install Python and tools
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv python3-dev \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# Install all required Python packages globally for all models
RUN uv sync 
CMD [ "bash" ]

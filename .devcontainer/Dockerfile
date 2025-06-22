# ---------- Stage 1: Base with PyTorch & CUDA ----------
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base

# ---------- Stage 2: Devcontainer features ----------
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04 AS dev-base

# Copy CUDA-enabled base (includes Python and PyTorch) into dev container
COPY --from=base / / 

# ---------- Stage 3: Final setup ----------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dev tools and Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    nano curl wget less git sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        polars numpy pandas \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-${USERNAME} \
    && chmod 0440 /etc/sudoers.d/99-${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

CMD ["bash"]

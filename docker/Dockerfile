# =========================
# Stage: base for CPU only
# =========================
FROM python:3.11-slim AS base-cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl gnupg ca-certificates \
      ffmpeg libgl1 libglib2.0-0 libgtk2.0-dev \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS（不要なら削除可）
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Poetry は 3.11 の pip で入る
RUN pip install --no-cache-dir "poetry==1.8.2"


# =========================
# Stage: base for CUDA
# =========================
# CUDA 12.1 + cuDNN 8（※12.1 系は cuDNN 8 が正）
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# Python3.11 を既定化（3.10 を使わない）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common curl gnupg ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils \
      git ffmpeg libgl1 libglib2.0-0 libgtk2.0-dev build-essential && \
    # 3.11 用の pip を用意
    python3.11 -m ensurepip --upgrade && \
    # python / python3 / pip / pip3 を 3.11 に向ける
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Node.js LTS（不要なら削除可）
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Poetry は 3.11 でインストール（これが重要）
RUN python3.11 -m pip install --no-cache-dir "poetry==1.8.2"


# =========================
# Stage: app for CPU
# =========================
FROM base-cpu AS app-cpu
WORKDIR /app

# 依存（キャッシュ最適化）
COPY poetry.lock pyproject.toml /app/

# システムに直接インストール（仮想環境は作らない）
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 8888
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]


# =========================
# Stage: app for CUDA
# =========================
FROM base-cuda AS app-cuda
WORKDIR /app

COPY poetry.lock pyproject.toml /app/

# ここで確実に 3.11 系を使う（上で既定化済み）
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# CUDA 12.1 用の PyTorch を明示（CPU 版で上書きされないように）
RUN poetry run pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

EXPOSE 8888
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

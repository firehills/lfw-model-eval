# GPU variant: CUDA 12 + cuDNN 9 runtime for onnxruntime-gpu (CUDAExecutionProvider).
# Requires an NVIDIA driver on the Windows host and Docker GPU support in WSL2.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev libgl1 libglib2.0-0 git sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && python3 -m venv /opt/venv \
    && chown -R ubuntu:ubuntu /opt/venv

COPY requirements.txt requirements-dev.txt requirements-gpu.txt /tmp/face-eval-requirements/
# onnxruntime and onnxruntime-gpu share a module name, so swap them explicitly.
RUN python -m pip install --no-cache-dir -r /tmp/face-eval-requirements/requirements-dev.txt \
    && python -m pip uninstall -y onnxruntime \
    && python -m pip install --no-cache-dir -r /tmp/face-eval-requirements/requirements-gpu.txt \
    && python -m pip check \
    && chown -R ubuntu:ubuntu /opt/venv

USER ubuntu
WORKDIR /workspaces/lfw-model-eval

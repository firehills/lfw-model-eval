FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Only Debian packages are needed; inherited optional repositories may have expired keys.
RUN apt-get -o Dir::Etc::sourcelist=sources.list.d/debian.sources \
        -o Dir::Etc::sourceparts=- update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/face-eval-requirements/
RUN python -m pip install --no-cache-dir -r /tmp/face-eval-requirements/requirements-dev.txt \
    && python -m pip check

WORKDIR /workspaces/lfw-model-eval

# base image
FROM ubuntu:22.04

WORKDIR /app

# Add Google Cloud CLI and gcsfuse (EL10)
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl lsb-release git wget && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    apt clean

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG EXTRA=tpu

COPY pyproject.toml envs/
COPY .python-version envs/

# Convert "data dev cuda" → "--extra data --extra dev --extra cuda"
RUN cd envs && \
    UV_CACHE_DIR=./tmp uv sync --extra ${EXTRA} && \
    rm -rf ./tmp

ENV PATH="/app/envs/.venv/bin:$PATH"

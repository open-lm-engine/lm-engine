# base image
FROM registry.access.redhat.com/ubi9/ubi

WORKDIR /app

# install git
RUN dnf install -y git && dnf clean all

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# install TPU dependencies
RUN git clone --recurse-submodules https://github.com/open-lm-engine/lm-engine && \
    cd lm-engine && \
    uv sync --extra tpu && \
    cd .. && \
    rm -rf lm-engine

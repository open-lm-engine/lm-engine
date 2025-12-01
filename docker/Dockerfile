# base image
FROM registry.access.redhat.com/ubi9/ubi

WORKDIR /app

# install git
RUN dnf install -y git && dnf clean all

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG EXTRA=tpu

# install TPU dependencies
RUN git clone --recurse-submodules https://github.com/open-lm-engine/lm-engine && \
    cd lm-engine && \
    git checkout trainium && \
    uv sync --extra ${EXTRA}

ENV PATH="/app/lm-engine/.venv/bin:$PATH"

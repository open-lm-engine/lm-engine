# base image
FROM registry.access.redhat.com/ubi9/ubi

WORKDIR /app

# Add Google Cloud CLI repo (EL10)
RUN printf "%s\n" \
    "[google-cloud-cli]" \
    "name=Google Cloud CLI" \
    "baseurl=https://packages.cloud.google.com/yum/repos/cloud-sdk-el10-x86_64" \
    "enabled=1" \
    "gpgcheck=1" \
    "repo_gpgcheck=0" \
    "gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key-v10.gpg" \
    > /etc/yum.repos.d/google-cloud-sdk.repo

# Install gcloud and git
RUN dnf install -y google-cloud-cli git && dnf clean all

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG EXTRA=data

# install dependencies
RUN git clone --recurse-submodules https://github.com/open-lm-engine/lm-engine && \
    cd lm-engine && \
    git checkout trainium && \
    uv sync --extra ${EXTRA}

ENV PATH="/app/lm-engine/.venv/bin:$PATH"

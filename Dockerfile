FROM python:3.8-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/

RUN apt-get update --allow-releaseinfo-change \
    && apt-get install --reinstall -y debian-archive-keyring \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && apt-get update --allow-releaseinfo-change \
    && apt-get install -y --no-install-recommends \
        libatlas-base-dev \
        liblapack-dev \
        build-essential \
        gfortran \
        gcc \
        curl \
        libclang-dev \
        ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.72.1 -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && export RUSTUP_TOOLCHAIN=1.72.1 \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

ENV PYTHONPATH=/app

CMD ["python", "scripts/train.py"]

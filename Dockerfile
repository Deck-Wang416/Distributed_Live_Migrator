FROM python:3.8-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update --allow-insecure-repositories \
    && apt-get install -y --no-install-recommends \
        libatlas-base-dev \
        liblapack-dev \
        build-essential \
        gfortran \
        gcc \
        curl \
        libclang-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.72.1 --profile minimal -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && export RUSTUP_TOOLCHAIN=1.72.1 \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV PATH="/root/.cargo/bin:${PATH}"

ENV PYTHONPATH=/app

CMD ["python", "scripts/train.py"]

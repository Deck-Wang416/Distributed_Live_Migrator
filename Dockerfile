FROM python:3.8-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    liblapack-dev \
    build-essential \
    gfortran \
    gcc \
    curl \
    libclang-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y \
    && . $HOME/.cargo/env \
    && pip install --upgrade pip \
    && pip install tokenizers==0.10.3 \
    && pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "scripts/train.py"]

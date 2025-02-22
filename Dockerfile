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
    rustc \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "scripts/train.py"]

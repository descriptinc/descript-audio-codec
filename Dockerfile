FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    gcc \
    g++ \
    build-essential \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install .

# Training/metrics dependencies used by scripts/train.py.
# numpy<2 is required because pesq 0.0.4 builds a NumPy C extension that is
# not compatible with NumPy 2.x in this environment.
RUN pip install --no-cache-dir \
    "numpy<2" \
    "transformers==4.36.2" \
    "pesq" \
    "openai-whisper" \
    "jiwer" \
    "typing-extensions>=4.14.1"

RUN python3 -m dac download

# Runtime note: multi-GPU training with many DataLoader workers needs more
# shared memory than Docker's default. Start the container with something like:
#   docker run --gpus all --shm-size=64g ...
# or:
#   docker run --gpus all --ipc=host ...
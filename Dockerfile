FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY . /app
WORKDIR /app

RUN apt update && apt install -y git curl wget ca-certificates

RUN pip install .
RUN pip install "numpy<2" "transformers==4.36.2"

RUN python3 -m dac download
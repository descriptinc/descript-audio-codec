FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY . /app
WORKDIR /app

RUN apt update && apt install -y git
# install the package
RUN pip install .

# cache the model
RUN python3 -m dac download

# Descript Audio Codec (.dac): High-Fidelity Audio Compression with Improved RVQGAN

This repository contains training and inference scripts
for the Descript Audio Codec (.dac), a high fidelity general
neural audio codec, introduced in the paper titled **High-Fidelity Audio Compression with Improved RVQGAN**.

![](https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-16x16.png) [arXiv Paper: High-Fidelity Audio Compression with Improved RVQGAN
](http://arxiv.org/abs/2306.06546) <br>
📈 [Demo Site](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5)<br>
⚙ [Model Weights](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth)

👉 With Descript Audio Codec, you can compress **44.1 KHz audio** into discrete codes at a **low 8 kbps bitrate**.  <br>
🤌 That's approximately **90x compression** while maintaining exceptional fidelity and minimizing artifacts.  <br>
💪 Our universal model works on all domains (speech, environment, music, etc.), making it widely applicable to generative modeling of all audio.  <br>
👌 It can be used as a drop-in replacement for EnCodec for all audio language modeling applications (such as AudioLMs, MusicLMs, MusicGen, etc.) <br>

<p align="center">
<img src="./assets/comparsion_stats.png" alt="Comparison of compressions approaches. Our model achieves a higher compression factor compared to all baseline methods. Our model has a ~90x compression factor compared to 32x compression factor of EnCodec and 64x of SoundStream. Note that we operate at a target bitrate of 8 kbps, whereas EnCodec operates at 24 kbps and SoundStream at 6 kbps. We also operate at 44.1 kHz, whereas EnCodec operates at 48 kHz and SoundStream operates at 24 kHz." width=35%></p>


## Usage

### Installation
```
pip install descript-audio-codec
```
OR

```
pip install git+https://github.com/descriptinc/descript-audio-codec
```

### Weights
Weights are released as part of this repo under MIT license.
We release weights for models that can natively support 16 kHz, 24kHz, and 44.1kHz sampling rates.
Weights are automatically downloaded when you first run `encode` or `decode` command. You can cache them using one of the following commands
```bash
python3 -m dac download # downloads the default 44kHz variant
python3 -m dac download --model_type 44khz # downloads the 44kHz variant
python3 -m dac download --model_type 24khz # downloads the 24kHz variant
python3 -m dac download --model_type 16khz # downloads the 16kHz variant
```
We provide a Dockerfile that installs all required dependencies for encoding and decoding. The build process caches the default model weights inside the image. This allows the image to be used without an internet connection. [Please refer to instructions below.](#docker-image)


### Compress audio
```
python3 -m dac encode /path/to/input --output /path/to/output/codes
```

This command will create `.dac` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac encode --help`
for more options.

### Reconstruct audio from compressed codes
```
python3 -m dac decode /path/to/output/codes --output /path/to/reconstructed_input
```

This command will create `.wav` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac decode --help`
for more options.

### Programmatic Usage
```py
import dac
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to('cuda')

# Load audio signal file
signal = AudioSignal('input.wav')

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode audio signal
y = model.decode(z)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write('output.wav')
```

### Docker image
We provide a dockerfile to build a docker image with all the necessary
dependencies.
1. Building the image.
    ```
    docker build -t dac .
    ```
2. Using the image.

    Usage on CPU:
    ```
    docker run dac <command>
    ```

    Usage on GPU:
    ```
    docker run --gpus=all dac <command>
    ```

    `<command>` can be one of the compression and reconstruction commands listed
    above. For example, if you want to run compression,

    ```
    docker run --gpus=all dac python3 -m dac encode ...
    ```


## Training
The baseline model configuration can be trained using the following commands.

### Pre-requisites
Please install the correct dependencies
```
pip install -e ".[dev]"
```

## Environment setup

We have provided a Dockerfile and docker compose setup that makes running experiments easy.

To build the docker image do:

```
docker compose build
```

Then, to launch a container, do:

```
docker compose run -p 8888:8888 -p 6006:6006 dev
```

The port arguments (`-p`) are optional, but useful if you want to launch a Jupyter and Tensorboard instances within the container. The
default password for Jupyter is `password`, and the current directory
is mounted to `/u/home/src`, which also becomes the working directory.

Then, run your training command.


### Single GPU training
```
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/baseline/
```

### Multi GPU training
```
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node gpu scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/baseline/
```

## Testing
We provide two test scripts to test CLI + training functionality. Please
make sure that the trainig pre-requisites are satisfied before launching these
tests. To launch these tests please run
```
python -m pytest tests
```

## Results

<p align="left">
<img src="./assets/objective_comparisons.png" width=75%></p>

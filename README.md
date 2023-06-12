# Descript Audio Codec (.dac)

<!-- ![](https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png) -->


This repository contains training and inference scripts
for the Descript Audio Codec (.dac), a high fidelity general
neural audio codec.


## Usage

### Installation
```
git clone https://github.com/descriptinc/descript-audio-codec
cd descript-audio-codec
pip install .
```

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

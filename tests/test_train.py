"""
Tests for CLI.
"""
import os
import shlex
import subprocess
from pathlib import Path

import argbind
import numpy as np
from audiotools import AudioSignal

from dac.__main__ import run


def make_fake_data(data_dir=Path(__file__).parent / "assets"):
    data_dir.mkdir(exist_ok=True, parents=True)
    input_dir = data_dir / "input"
    input_dir.mkdir(exist_ok=True, parents=True)

    for i in range(100):
        signal = AudioSignal(np.random.randn(44_100 * 5), 44_100)
        signal.write(input_dir / f"sample_{i}.wav")
    return input_dir


def make_fake_data_tree():
    data_dir = Path(__file__).parent / "assets"

    for relative_dir in [
        "train/speech",
        "train/music",
        "train/env",
        "val/speech",
        "val/music",
        "val/env",
        "test/speech",
        "test/music",
        "test/env",
    ]:
        leaf_dir = data_dir / relative_dir
        leaf_dir.mkdir(exist_ok=True, parents=True)
        make_fake_data(leaf_dir)
    return {
        split: {
            key: [str(data_dir / f"{split}/{key}")]
            for key in ["speech", "music", "env"]
        }
        for split in ["train", "val", "test"]
    }


def setup_module(module):
    # Make fake dataset dir
    input_datasets = make_fake_data_tree()
    repo_root = Path(__file__).parent.parent

    # Load baseline conf and modify it for testing
    conf = argbind.load_args(repo_root / "conf" / "ablations" / "baseline.yml")

    for key in ["train", "val", "test"]:
        conf[f"{key}/build_dataset.folders"] = input_datasets[key]
    conf["num_iters"] = 1
    conf["val/AudioDataset.n_examples"] = 1
    conf["val_idx"] = [0]
    conf["val_batch_size"] = 1

    argbind.dump_args(conf, Path(__file__).parent / "assets" / "conf.yml")


def teardown_module(module):
    repo_root = Path(__file__).parent.parent
    # Remove fake dataset dir
    subprocess.check_output(["rm", "-rf", f"{repo_root}/tests/assets"])
    subprocess.check_output(["rm", "-rf", f"{repo_root}/tests/runs"])


def test_single_gpu_train():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    repo_root = Path(__file__).parent.parent
    args = shlex.split(
        f"python {repo_root}/scripts/train.py --args.load {repo_root}/tests/assets/conf.yml --save_path {repo_root}/tests/runs/baseline"
    )
    subprocess.check_output(args, env=env)


def test_multi_gpu_train():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    repo_root = Path(__file__).parent.parent
    args = shlex.split(
        f"torchrun --nproc_per_node gpu {repo_root}/scripts/train.py --args.load {repo_root}/tests/assets/conf.yml --save_path {repo_root}/tests/runs/baseline_multigpu"
    )
    subprocess.check_output(args, env=env)

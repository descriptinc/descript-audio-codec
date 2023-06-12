import os
import pathlib
import shutil
from collections import defaultdict
from typing import Tuple

import argbind
import numpy as np
import tqdm
from audiotools import util


@argbind.bind()
def split(
    audio_files, ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
):
    assert sum(ratio) == 1.0
    util.seed(seed)

    idx = np.arange(len(audio_files))
    np.random.shuffle(idx)

    b = np.cumsum([0] + list(ratio)) * len(idx)
    b = [int(_b) for _b in b]
    train_idx = idx[b[0] : b[1]]
    val_idx = idx[b[1] : b[2]]
    test_idx = idx[b[2] :]

    audio_files = np.array(audio_files)
    train_files = audio_files[train_idx]
    val_files = audio_files[val_idx]
    test_files = audio_files[test_idx]

    return train_files, val_files, test_files


def assign(val_split, test_split):
    def _assign(value):
        if value in val_split:
            return "val"
        if value in test_split:
            return "test"
        return "train"

    return _assign


DAPS_VAL = ["f2", "m2"]
DAPS_TEST = ["f10", "m10"]


@argbind.bind(without_prefix=True)
def process(
    dataset: str = "daps",
    daps_subset: str = "",
):
    get_split = None
    get_value = lambda path: path

    data_path = pathlib.Path("/data")
    dataset_path = data_path / dataset
    audio_files = util.find_audio(dataset_path)

    if dataset == "daps":
        get_split = assign(DAPS_VAL, DAPS_TEST)
        get_value = lambda path: (str(path).split("/")[-1].split("_", maxsplit=4)[0])
        audio_files = [
            x
            for x in util.find_audio(dataset_path)
            if daps_subset in str(x) and "breaths" not in str(x)
        ]

    if get_split is None:
        _, val, test = split(audio_files)
        get_split = assign(val, test)

    splits = defaultdict(list)
    for x in audio_files:
        _split = get_split(get_value(x))
        splits[_split].append(x)

    with util.chdir(dataset_path):
        for k, v in splits.items():
            v = sorted(v)
            print(f"Processing {k} in {dataset_path} of length {len(v)}")
            for _v in tqdm.tqdm(v):
                tgt_path = pathlib.Path(
                    str(_v).replace(str(dataset_path), str(dataset_path / k))
                )
                tgt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_v, tgt_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        process()

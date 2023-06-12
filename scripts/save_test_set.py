import csv
from pathlib import Path

import argbind
import torch
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import Accelerator

import scripts.train as train


@torch.no_grad()
def process(batch, accel, test_data):
    batch = util.prepare_batch(batch, accel.device)
    signal = test_data.transform(batch["signal"].clone(), **batch["transform_args"])
    return signal.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def save_test_set(args, accel, sample_rate: int = 44100, output: str = "samples/input"):
    tracker = Tracker()
    with argbind.scope(args, "test"):
        test_data = train.build_dataset(sample_rate)

    global process
    process = tracker.track("process", len(test_data))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "input").mkdir(parents=True, exist_ok=True)
    with open(output / "metadata.csv", "w") as csvfile:
        keys = ["path", "original"]
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()

        with tracker.live:
            for i in range(len(test_data)):
                signal = process(test_data[i], accel, test_data)
                input_path = output.parent / "input" / f"sample_{i}.wav"
                metadata = {
                    "path": str(input_path),
                    "original": str(signal.path_to_input_file),
                }
                writer.writerow(metadata)
                signal.write(input_path)
            tracker.done("test", f"N={len(test_data)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            save_test_set(args, accel)

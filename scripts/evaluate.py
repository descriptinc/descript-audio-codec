import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import losses


@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss


def get_metrics(signal_path, recons_path, state):
    output = {}
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    for sr in [22050, 44100]:
        x = signal.clone().resample(sr)
        y = recons.clone().resample(sr)
        k = "22k" if sr == 22050 else "44k"
        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y),
                f"stft-{k}": state.stft_loss(x, y),
                f"waveform-{k}": state.waveform_loss(x, y),
                f"sisdr-{k}": state.sisdr_loss(x, y),
                f"visqol-audio-{k}": metrics.quality.visqol(x, y),
                f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech"),
            }
        )
    output["path"] = signal.path_to_file
    output.update(signal.metadata)
    return output


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "samples/input",
    output: str = "samples/output",
    n_proc: int = 50,
):
    tracker = Tracker()

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    sisdr_loss = losses.SISDRLoss()

    state = State(
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        sisdr_loss=sisdr_loss,
    )

    audio_files = util.find_audio(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        o = future.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    futures = []
    with tracker.live:
        with open(output / "metrics.csv", "w") as csvfile:
            with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
                for i in range(len(audio_files)):
                    future = pool.submit(
                        get_metrics, audio_files[i], output / audio_files[i].name, state
                    )
                    futures.append(future)

                keys = list(futures[0].result().keys())
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()

                for future in futures:
                    record(future, writer)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()

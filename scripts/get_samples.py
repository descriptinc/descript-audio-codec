from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import Accelerator
from train import DAC

from dac.compare.encodec import Encodec

Encodec = argbind.bind(Encodec)


def load_state(
    accel: Accelerator,
    tracker: Tracker,
    save_path: str,
    tag: str = "latest",
    load_weights: bool = False,
    model_type: str = "dac",
    bandwidth: float = 24.0,
):
    kwargs = {
        "folder": f"{save_path}/{tag}",
        "map_location": "cpu",
        "package": not load_weights,
    }
    tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")

    if model_type == "dac":
        generator, _ = DAC.load_from_folder(**kwargs)
    elif model_type == "encodec":
        generator = Encodec(bandwidth=bandwidth)

    generator = accel.prepare_model(generator)
    return generator


@torch.no_grad()
def process(signal, accel, generator, **kwargs):
    signal = signal.to(accel.device)
    recons = generator(signal.audio_data, signal.sample_rate, **kwargs)["audio"]
    recons = AudioSignal(recons, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    accel,
    path: str = "ckpt",
    input: str = "samples/input",
    output: str = "samples/output",
    model_type: str = "dac",
    model_tag: str = "latest",
    bandwidth: float = 24.0,
    n_quantizers: int = None,
):
    tracker = Tracker(log_file=f"{path}/eval.txt", rank=accel.local_rank)
    generator = load_state(
        accel,
        tracker,
        save_path=path,
        model_type=model_type,
        bandwidth=bandwidth,
        tag=model_tag,
    )
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers} if model_type == "dac" else {}

    audio_files = util.find_audio(input)

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for i in range(len(audio_files)):
            signal = AudioSignal(audio_files[i])
            recons = process(signal, accel, generator, **kwargs)
            recons.write(output / audio_files[i].name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            get_samples(accel)

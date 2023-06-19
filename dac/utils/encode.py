import math
import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

import dac
from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
@torch.inference_mode()
def process(
    signal: AudioSignal, device: str, generator: torch.nn.Module, **kwargs
) -> dict:
    """Encode an audio signal. The signal is chunked into overlapping windows
    and encoded one by one.

    Parameters
    ----------
    signal : AudioSignal
        Input signal to encode
    device : str
        Device to use
    generator : torch.nn.Module
        Generator to encode with

    Returns
    -------
    dict
        Dictionary of artifacts with the following keys:
        - codes: the quantized codes
        - metadata: dictionary with following keys
            - original_db: the loudness of the input signal
            - overlap_hop_duration: the hop duration of the overlap window
            - original_length: the original length of the input signal
            - is_overlap: whether the input signal was overlapped
            - batch_size: the batch size of the input signal
            - channels: the number of channels of the input signal
            - original_sr: the original sample rate of the input signal

    """
    if isinstance(generator, torch.nn.DataParallel):
        generator = generator.module

    original_sr = signal.sample_rate

    # Resample input
    audio_signal = signal.ffmpeg_resample(generator.sample_rate)

    original_length = audio_signal.signal_length
    input_db = audio_signal.ffmpeg_loudness()

    # Set variables
    sr = audio_signal.sample_rate
    overlap_win_duration = 5.0
    overlap_hop_ratio = 0.5

    # Fix overlap window so that it's divisible by 4 in # of samples
    overlap_win_duration = ((overlap_win_duration * sr) // 4) * 4
    overlap_win_duration = overlap_win_duration / sr
    overlap_hop_duration = overlap_win_duration * overlap_hop_ratio
    do_overlap_and_add = audio_signal.signal_duration > overlap_win_duration

    # TODO (eeishaan): Remove this when correct caching logic is implemented and
    # overlap of codes is minimal
    do_overlap_and_add = False

    # Sanitize input
    audio_signal.normalize(-16)
    audio_signal.ensure_max_of_audio()

    nb, nac, nt = audio_signal.audio_data.shape
    audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)

    if do_overlap_and_add:
        pad_length = (
            math.ceil(audio_signal.signal_duration / overlap_win_duration)
            * overlap_win_duration
        )
        audio_signal.zero_pad_to(int(pad_length * sr))
        audio_signal = audio_signal.collect_windows(
            overlap_win_duration, overlap_hop_duration
        )

    codebook_indices = []
    for i in range(audio_signal.batch_size):
        signal_from_batch = AudioSignal(
            audio_signal.audio_data[i, ...], audio_signal.sample_rate
        )
        signal_from_batch.to(device)
        codes = generator.encode(
            signal_from_batch.audio_data, signal_from_batch.sample_rate, **kwargs
        )["codes"].cpu()
        codebook_indices.append(codes)

    codebook_indices = torch.cat(codebook_indices, dim=0)

    return {
        "codes": codebook_indices.numpy().astype(np.uint16),
        "metadata": {
            "original_db": input_db,
            "overlap_hop_duration": overlap_hop_duration,
            "original_length": original_length,
            "is_overlap": do_overlap_and_add,
            "batch_size": nb,
            "channels": nac,
            "original_sr": original_sr,
        },
    }


@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = dac.__model_version__,
    n_quantizers: int = None,
    device: str = "cuda",
):
    generator = load_model(
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    # Find all audio files in input path
    input = Path(input)
    audio_files = util.find_audio(input)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(audio_files)), desc="Encoding files"):
        # Load file
        signal = AudioSignal(audio_files[i])

        # Encode audio to .dac format
        artifacts = process(signal, device, generator, **kwargs)

        # Compute output path
        relative_path = audio_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = audio_files[i]
        output_name = relative_path.with_suffix(".dac").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, "wb") as f:
            np.save(f, artifacts)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()

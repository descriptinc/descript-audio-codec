import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from tqdm import tqdm

import dac
from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
@torch.inference_mode()
def process(
    artifacts: dict,
    device: str,
    generator: torch.nn.Module,
    preserve_sample_rate: bool,
) -> AudioSignal:
    """Decode encoded audio. The `artifacts` contain codes from chunked windows
    of the original audio signal. The codes are decoded one by one and windows are trimmed and concatenated together to form the final output.

    Parameters
    ----------
    artifacts : dict
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
    device : str
        Device to use
    generator : torch.nn.Module
        Generator to decode with.
    preserve_sample_rate : bool
        If True, return audio will have the same sample rate as the original
        encoded audio. If False, return audio will have the sample rate of the
        generator.

    Returns
    -------
    AudioSignal
    """
    if isinstance(generator, torch.nn.DataParallel):
        generator = generator.module
    audio_signal = AudioSignal(
        artifacts["codes"].astype(np.int64), generator.sample_rate
    )
    metadata = artifacts["metadata"]

    # Decode chunks
    output = []
    for i in range(audio_signal.batch_size):
        signal_from_batch = AudioSignal(
            audio_signal.audio_data[i, ...], audio_signal.sample_rate, device=device
        )
        z_q = generator.quantizer.from_codes(signal_from_batch.audio_data)[0]
        audio = generator.decode(z_q)["audio"].cpu()
        output.append(audio)

    output = torch.cat(output, dim=0)
    output_signal = AudioSignal(output, generator.sample_rate)

    # Overlap and add
    if metadata["is_overlap"]:
        boundary = int(metadata["overlap_hop_duration"] * generator.sample_rate / 2)
        # remove window overlap
        output_signal.trim(boundary, boundary)
        output_signal.audio_data = output_signal.audio_data.reshape(
            metadata["batch_size"], metadata["channels"], -1
        )
        # remove padding
        output_signal.trim(boundary, boundary)

    # Restore loudness and truncate to original length
    output_signal.ffmpeg_loudness()
    output_signal = output_signal.normalize(metadata["original_db"])
    output_signal.truncate_samples(metadata["original_length"])

    if preserve_sample_rate:
        output_signal = output_signal.ffmpeg_resample(metadata["original_sr"])

    return output_signal.to("cpu")


@argbind.bind(group="decode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def decode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = dac.__model_version__,
    preserve_sample_rate: bool = False,
    device: str = "cuda",
):
    generator = load_model(
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()

    # Find all .dac files in input directory
    _input = Path(input)
    input_files = list(_input.glob("**/*.dac"))

    # If input is a .dac file, add it to the list
    if _input.suffix == ".dac":
        input_files.append(_input)

    # Create output directory
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(input_files)), desc=f"Decoding files"):
        # Load file
        artifacts = np.load(input_files[i], allow_pickle=True)[()]

        # Reconstruct audio from codes
        recons = process(artifacts, device, generator, preserve_sample_rate)

        # Compute output path
        relative_path = input_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = input_files[i]
        output_name = relative_path.with_suffix(".wav").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        recons.write(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        decode()

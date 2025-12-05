#!/usr/bin/env python3
"""
Evaluate a DAC model over a dataset of WAV files by computing PESQ, SIM, and WER
between the original and encoded-decoded audio.

Usage: python scripts/compute_metrics.py --model_path <path> --dataset_path <path> [--device cuda|cpu]

Notes:
- Iterates recursively over all .wav files under dataset_path
- Does not use autocast or write any output audio files
- Prints mean and 95% confidence interval for each metric
"""

from pathlib import Path
from typing import List, Tuple

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.data import transforms

from dac.model import DAC
from dac.utils.transforms import PowerNorm
from dac.metrics import SIM, PESQ, WER
from tqdm.auto import tqdm


def _to_mono_numpy(audio_tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor shaped [C, T] or [T] (optionally with batch dim) to mono numpy [T]."""
    # Remove batch dimension(s)
    audio = audio_tensor.detach().cpu().squeeze()
    if audio.ndim == 2:
        # Average channels to mono
        audio = audio.mean(dim=0)
    return audio.float().numpy()


def _mean_ci(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    n = int(arr.size)
    mean = float('nan')
    ci95 = float('nan')
    if n > 0:
        mean = float(arr.mean())
        if n > 1:
            ci95 = 1.96 * float(arr.std(ddof=1)) / np.sqrt(n)
    return mean, ci95


@argbind.bind(without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def evaluate_dataset(
    model_path: Path = Path(""),
    dataset_path: str = "",
    device: str = "",
):
    """Evaluate PESQ, SIM, and WER over a dataset of WAV files.

    Parameters
    ----------
    model_path : str
        Path to the DAC model checkpoint.
    dataset_path : str
        Directory to recursively search for .wav files.
    device : str
        Device to use; defaults to cuda if available else cpu.
    """
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model package
    model = DAC.load(model_path / "best/dac/package.pth")
    model.to(device)
    model.eval()

    # Metrics
    sim_metric = SIM(device=device)
    pesq_metric = PESQ()
    wer_metric = WER(device=device)

    # Match training postprocess: PowerNorm (apply explicitly)
    _power_norm = PowerNorm(db=-16.0)

    sim_scores: List[float] = []
    pesq_scores: List[float] = []
    wer_scores: List[float] = []

    wav_files = sorted(Path(dataset_path).rglob("*.wav"))
    for wav_path in tqdm(wav_files):
        # Load and resample to model's sample rate for encode/decode
        signal = AudioSignal(str(wav_path))
        if signal.sample_rate != model.sample_rate:
            signal.resample(model.sample_rate)
        signal.to(device)

        # Apply same postprocess used in training
        signal = _power_norm._transform(signal)

        # Encode/decode without autocast
        z = model.encode(signal.audio_data)
        # Older models return a tuple; use the first element as z.
        if isinstance(z, tuple):
            z = z[0]
        y = model.decode(z)

        # Prepare numpy vectors
        original_np = _to_mono_numpy(signal.audio_data)
        decoded_np = _to_mono_numpy(y)

        # SIM (higher is better)
        sim_scores.append(sim_metric(original_np, decoded_np, sample_rate=model.sample_rate))

        # PESQ (higher is better)
        pesq_scores.append(pesq_metric(original_np, decoded_np, sample_rate=model.sample_rate))

        # WER (lower is better) using in-memory arrays
        wer_scores.append(wer_metric(original_np, decoded_np, sample_rate=model.sample_rate))

    # Aggregate and print results
    sim_mean, sim_ci = _mean_ci(sim_scores)
    pesq_mean, pesq_ci = _mean_ci(pesq_scores)
    wer_mean, wer_ci = _mean_ci(wer_scores)

    print(f"SIM = {sim_mean:.2f} ± {sim_ci:.2f}")
    print(f"PESQ = {pesq_mean:.2f} ± {pesq_ci:.2f}")
    print(f"WER = {wer_mean:.2f} ± {wer_ci:.2f}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate_dataset()



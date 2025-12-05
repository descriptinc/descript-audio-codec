import torch
import numpy as np
import argbind
from pathlib import Path
from audiotools import AudioSignal
from audiotools.data import transforms
from dac.model import DAC
from dac.utils.transforms import PowerNorm
from tqdm import tqdm

from dac.metrics.eval_utils import (
    compute_condition_number,
    compute_smoothness_curve,
    compute_locality_curve,
    visualize_covariance_matrix,
    visualize_smoothness_curve,
    visualize_locality_curve,
)


def _mean_ci(values):
    arr = np.array(values, dtype=float)
    mean = float('nan')
    ci95 = float('nan')
    if arr.size > 0:
        mean = float(arr.mean())
        if arr.size > 1:
            ci95 = 1.96 * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return mean, ci95


@argbind.bind(without_prefix=True)
def analyze_latent_perturbations(
    dataset_path: Path = Path(""),
    model_path: Path = Path(""),
    hop_length: int = 2048,
    n_mels: int = 64,
    window_before: int = 10,
    window_after: int = 10,
    max_batch_size: int = 16,
    sample_perturbations: str = "",
):
    """Analyze how perturbations to DAC latents affect mel-spectrogram reconstruction over a dataset.
    
    Args:
        dataset_path (Path): Root directory. Recursively finds all .wav files beneath this path.
        model_path (Path): Path to model checkpoint file
        hop_length (int): Hop length for mel spectrogram
        n_mels (int): Number of mel bins
        window_before (int): Number of frames before perturbation to consider
        window_after (int): Number of frames after perturbation to consider
        max_batch_size (int): Maximum batch size for perturbation calculations
        sample_perturbations (str): Comma-separated list of perturbation magnitudes to save audio samples for
    """
    # Device handling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory
    output_dir = Path(dataset_path) if str(dataset_path) != "" else Path(".")
    
    # Parse sample perturbations
    sample_magnitudes = [float(x.strip()) for x in sample_perturbations.split(',') if x.strip()] if sample_perturbations.strip() else []
    
    # Load model package and explicitly move to specified device
    model = DAC.load(model_path / "best/dac/package.pth")
    model.to(device)
    model.eval()
    
    # Prepare transforms (match training postprocess)
    _power_norm = PowerNorm(db=-16.0)

    # Collect all wav files
    wav_files = sorted(Path(dataset_path).rglob("*.wav")) if str(dataset_path) != "" else []

    # Aggregation containers
    baseline_mcd_list = []
    cond_number_list = []
    cov_accum = None

    perturbation_magnitudes = np.logspace(-2, 0, 100)
    smoothness_sum = np.zeros_like(perturbation_magnitudes, dtype=float)
    smoothness_count = 0

    relative_distances = list(range(-window_before, window_after + 1))
    locality_sum = np.zeros(len(relative_distances), dtype=float)
    locality_count = 0

    for wav_path in tqdm(wav_files, desc="Files"):
        # Load and process audio
        signal = AudioSignal(str(wav_path))
        if signal.sample_rate != model.sample_rate:
            signal.resample(model.sample_rate)
        signal.to(device)
        
        # Apply training postprocess
        signal = _power_norm._transform(signal)
        # Ensure model padding behavior still applies before encoding
        signal.audio_data = model.preprocess(signal.audio_data, signal.sample_rate)
        
        # Get original latents
        with torch.no_grad():
            latents_orig = model.encode(signal.audio_data)
            if isinstance(latents_orig, tuple):
                latents_orig = latents_orig[0]
        
        # Compute condition number and covariance for this file
        cond_number, cov_matrix, eigenvals, eigenvecs = compute_condition_number(latents_orig)
        cond_number_list.append(cond_number)
        
        # Accumulate normalized covariance
        cov_normalized = cov_matrix / np.trace(cov_matrix) * cov_matrix.shape[0]
        if cov_accum is None:
            cov_accum = np.zeros_like(cov_normalized)
        cov_accum += cov_normalized
        
        # Compute smoothness curve for this file
        mcd_errors, baseline_mcd = compute_smoothness_curve(
            model, signal, latents_orig, perturbation_magnitudes,
            n_mels=n_mels, hop_length=hop_length, max_batch_size=max_batch_size
        )
        baseline_mcd_list.append(baseline_mcd)
        smoothness_sum += mcd_errors
        smoothness_count += 1
        
        # Optional: Generate perturbed audio samples
        if sample_magnitudes:
            cov_sqrt = torch.tensor(
                eigenvecs @ np.diag(np.sqrt(eigenvals)), 
                device=device, 
                dtype=torch.float32
            )
            
            for start_idx in range(0, len(sample_magnitudes), max_batch_size):
                end_idx = min(start_idx + max_batch_size, len(sample_magnitudes))
                batch_magnitudes = sample_magnitudes[start_idx:end_idx]
                if len(batch_magnitudes) == 0:
                    break
                    
                latents_batch = latents_orig.repeat(len(batch_magnitudes), 1, 1)
                latents_batch += torch.einsum(
                    'ij,bjk->bik', 
                    cov_sqrt, 
                    torch.randn(len(batch_magnitudes), latents_batch.shape[1], 
                               latents_batch.shape[2], device=device)
                ) * torch.tensor(batch_magnitudes, device=device).view(-1, 1, 1)
                
                with torch.no_grad():
                    audio_batch = model.decode(latents_batch)
                    
                for i in range(len(batch_magnitudes)):
                    perturbed_audio = AudioSignal(audio_batch[i:i+1].cpu(), sample_rate=signal.sample_rate)
                    perturbed_audio.write(output_dir / f"{Path(wav_path).stem}_{model_path.name}_{start_idx + i}.wav")
        
        # Compute locality curve for this file
        relative_times, mcd_values = compute_locality_curve(
            model, signal, latents_orig, 
            window_before=window_before, window_after=window_after,
            n_mels=n_mels, hop_length=hop_length, max_batch_size=max_batch_size
        )
        locality_sum += mcd_values
        locality_count += 1

    # Aggregate statistics
    cov_mean = cov_accum / max(len(wav_files), 1)
    cond_mean, cond_ci = _mean_ci(cond_number_list)
    baseline_mean, baseline_ci = _mean_ci(baseline_mcd_list)
    smoothness_mean = smoothness_sum / max(smoothness_count, 1)
    locality_mean = locality_sum / max(locality_count, 1)
    
    # Plot mean covariance matrix
    visualize_covariance_matrix(
        cov_mean, cond_mean,
        output_path=output_dir / f"covariance_{model_path.name}_aggregate.svg"
    )
    
    # Plot smoothness curve
    visualize_smoothness_curve(
        perturbation_magnitudes, smoothness_mean, baseline_mean,
        output_path=output_dir / f"smoothness_{model_path.name}_aggregate.svg"  
    )
    
    # Plot locality curve
    visualize_locality_curve(
        relative_times, locality_mean,
        output_path=output_dir / f"locality_{model_path.name}_aggregate.svg"
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        analyze_latent_perturbations()
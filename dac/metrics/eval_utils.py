"""
Evaluation utilities for DAC models.
Contains functions for visualization and metrics computation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from audiotools import AudioSignal
from sklearn.decomposition import PCA
from scipy.fftpack import dct
import wandb
import os
import tempfile
import shutil


def compute_mcd(mel_spec1, mel_spec2):
    """
    Compute Mel-Cepstral Distortion (MCD) between two mel-spectrograms.
    
    Args:
        mel_spec1, mel_spec2: mel-spectrograms [n_mels, n_frames]
    
    Returns:
        MCD value in dB
    """
    min_length = min(mel_spec1.shape[-1], mel_spec2.shape[-1])
    return (10 / np.log(10)) * np.mean(np.sqrt(2 * np.sum((
        dct(np.log(mel_spec1[..., :min_length] + 1e-8), axis=0, norm='ortho') -
        dct(np.log(mel_spec2[..., :min_length] + 1e-8), axis=0, norm='ortho'))**2, axis=0)))


def compute_condition_number(latents):
    """
    Compute condition number of latent covariance matrix.
    
    Args:
        latents: torch.Tensor or numpy.ndarray of shape [batch, channels, time]
    
    Returns:
        tuple: (condition_number, covariance_matrix, eigenvalues, eigenvectors)
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.squeeze(0).cpu().numpy()
    
    # Compute covariance matrix
    cov_matrix = np.cov(latents)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals.real, 0)
    
    # Compute condition number
    cond_number = float(np.max(eigenvals) / (np.min(eigenvals) + 1e-10))
    
    return cond_number, cov_matrix, eigenvals, eigenvecs


def visualize_latents_with_pca(signal, latents, n_components=64, perform_pca=True, 
                               output_path=None, n_mels=64, hop_length=2048):
    """
    Visualize mel spectrogram and PCA components of DAC latents.
    
    Args:
        signal: AudioSignal object
        latents: torch.Tensor of shape [batch, channels, time]
        n_components: Number of PCA components to visualize
        perform_pca: Whether to perform PCA on latents
        output_path: Path to save the figure (if None, returns figure)
        n_mels: Number of mel bins
        hop_length: Hop length for mel spectrogram
    
    Returns:
        matplotlib.figure.Figure object if output_path is None
    """
    # Compute mel spectrogram
    mel_spec = signal.mel_spectrogram(
        n_mels=n_mels, 
        window_length=hop_length*2, 
        hop_length=hop_length
    ).squeeze().cpu().numpy()
    
    # Process latents
    latents_np = latents.squeeze().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
    
    if perform_pca and latents_np.shape[0] > n_components:
        pca = PCA(n_components=n_components)
        latents_np = pca.fit_transform(latents_np.T).T  # [n_components, n_frames]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Plot mel spectrogram
    ax1.imshow(
        np.log(mel_spec + 1e-8),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, signal.signal_duration, 0, mel_spec.shape[0]]
    )
    ax1.set_ylabel('Mel Bin')
    ax1.set_title('Mel Spectrogram')
    
    # Plot PCA components or raw latents
    ax2.imshow(
        latents_np,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
        extent=[0, signal.signal_duration, 0, latents_np.shape[0]]
    )
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('PCA Component' if perform_pca else 'Latent Dimension')
    ax2.set_title('DAC Latents' + (' (PCA)' if perform_pca else ''))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='svg' if str(output_path).endswith('.svg') else 'png', dpi=150)
        plt.close()
        return None
    else:
        return fig


def visualize_covariance_matrix(cov_matrix, condition_number, output_path=None):
    """
    Visualize covariance matrix with condition number.
    
    Args:
        cov_matrix: Numpy array covariance matrix
        condition_number: Condition number of the matrix
        output_path: Path to save the figure (if None, returns figure)
    
    Returns:
        matplotlib.figure.Figure object if output_path is None
    """
    # Normalize covariance matrix
    cov_normalized = cov_matrix / np.trace(cov_matrix) * cov_matrix.shape[0]
    
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cov_normalized, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Covariance / Trace * Dim')
    plt.title(f'Latent Covariance Matrix\nCondition Number: {condition_number:.1e}')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Latent Dimension')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='svg' if str(output_path).endswith('.svg') else 'png', dpi=150)
        plt.close()
        return None
    else:
        return fig


def compute_smoothness_curve(model, signal, latents_orig, perturbation_magnitudes, 
                            n_mels=64, hop_length=2048, max_batch_size=16):
    """
    Compute smoothness (robustness) curve for perturbations.
    
    Args:
        model: DAC model
        signal: Original AudioSignal
        latents_orig: Original latents
        perturbation_magnitudes: Array of perturbation magnitudes to test
        n_mels: Number of mel bins
        hop_length: Hop length for mel spectrogram
        max_batch_size: Maximum batch size for processing
    
    Returns:
        tuple: (mcd_errors, baseline_mcd)
    """
    device = latents_orig.device if isinstance(latents_orig, torch.Tensor) else 'cpu'
    
    # Compute baseline reconstruction
    with torch.no_grad():
        mel_orig = signal.mel_spectrogram(
            n_mels=n_mels, window_length=hop_length, hop_length=hop_length
        ).squeeze().cpu().numpy()
        
        mel_recons = AudioSignal(
            model.decode(latents_orig), 
            sample_rate=signal.sample_rate
        ).mel_spectrogram(
            n_mels=n_mels, window_length=hop_length, hop_length=hop_length
        ).squeeze().cpu().numpy()
    
    baseline_mcd = compute_mcd(mel_orig, mel_recons)
    
    # Compute covariance for perturbations
    _, _, eigenvals, eigenvecs = compute_condition_number(latents_orig)
    cov_sqrt = torch.tensor(
        eigenvecs @ np.diag(np.sqrt(eigenvals)), 
        device=device, 
        dtype=torch.float32
    )
    
    # Compute MCD for different perturbation magnitudes
    mcd_errors = []
    for start_idx in range(0, len(perturbation_magnitudes), max_batch_size):
        end_idx = min(start_idx + max_batch_size, len(perturbation_magnitudes))
        batch_magnitudes = perturbation_magnitudes[start_idx:end_idx]
        
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
            mel_pert = AudioSignal(
                audio_batch[i:i+1], 
                sample_rate=signal.sample_rate
            ).mel_spectrogram(
                n_mels=n_mels, window_length=hop_length, hop_length=hop_length
            ).squeeze().cpu().numpy()
            mcd_errors.append(compute_mcd(mel_orig, mel_pert))
    
    return np.array(mcd_errors), baseline_mcd


def visualize_smoothness_curve(perturbation_magnitudes, mcd_errors, baseline_mcd, output_path=None):
    """
    Visualize smoothness (robustness) curve.
    
    Args:
        perturbation_magnitudes: Array of perturbation magnitudes
        mcd_errors: Array of MCD errors for each magnitude
        baseline_mcd: Baseline MCD value
        output_path: Path to save the figure (if None, returns figure)
    
    Returns:
        matplotlib.figure.Figure object if output_path is None
    """
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(perturbation_magnitudes, mcd_errors, 'b-', linewidth=2)
    plt.axhline(y=baseline_mcd, color='r', linestyle='--', 
                label=f'Baseline MCD: {baseline_mcd:.2f} dB')
    plt.xlabel('Perturbation Magnitude')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(perturbation_magnitudes), max(perturbation_magnitudes))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Robustness to Latent Perturbations')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='svg' if str(output_path).endswith('.svg') else 'png', dpi=150)
        plt.close()
        return None
    else:
        return fig


def compute_locality_curve(model, signal, latents_orig, window_before=10, window_after=10, 
                          n_mels=64, hop_length=2048, max_batch_size=16):
    """
    Compute temporal locality curve for perturbations.
    
    Args:
        model: DAC model
        signal: Original AudioSignal
        latents_orig: Original latents
        window_before: Number of frames before perturbation to consider
        window_after: Number of frames after perturbation to consider
        n_mels: Number of mel bins
        hop_length: Hop length for mel spectrogram
        max_batch_size: Maximum batch size for processing
    
    Returns:
        tuple: (relative_times, mcd_values)
    """
    device = latents_orig.device if isinstance(latents_orig, torch.Tensor) else 'cpu'
    
    # Get encoder strides
    try:
        encoder_strides = model.encoder_strides
    except:
        encoder_strides = model.encoder_rates
    
    # Compute baseline reconstruction
    with torch.no_grad():
        mel_recons = AudioSignal(
            model.decode(latents_orig), 
            sample_rate=signal.sample_rate
        ).mel_spectrogram(
            n_mels=n_mels, window_length=hop_length, hop_length=hop_length
        ).squeeze().cpu().numpy()
    
    # Compute covariance for perturbations
    _, cov_matrix, eigenvals, eigenvecs = compute_condition_number(latents_orig)
    cov_sqrt = torch.tensor(
        eigenvecs @ np.diag(np.sqrt(eigenvals)), 
        device=device, 
        dtype=torch.float32
    )
    
    # Define relative distances and positions
    relative_distances = list(range(-window_before, window_after + 1))
    positions = list(range(window_before, latents_orig.shape[-1] - window_after))
    
    if len(positions) == 0:
        # If sequence too short, return zeros
        relative_times = [dist * hop_length / signal.sample_rate for dist in relative_distances]
        return np.array(relative_times), np.zeros(len(relative_distances))
    
    # Compute MCD for different relative positions
    mcd_values = np.zeros(len(relative_distances), dtype=float)
    
    for start_idx in range(0, len(positions), max_batch_size):
        end_idx = min(start_idx + max_batch_size, len(positions))
        batch_positions = positions[start_idx:end_idx]
        
        latents_batch = latents_orig.repeat(len(batch_positions), 1, 1)
        latents_batch[
            torch.arange(len(batch_positions), device=device), :, 
            torch.tensor(batch_positions, device=device)
        ] += torch.einsum(
            'ij,bj->bi', 
            cov_sqrt, 
            torch.randn(len(batch_positions), latents_batch.shape[1], device=device)
        )
        
        with torch.no_grad():
            audio_batch = model.decode(latents_batch)
        
        for i, pos in enumerate(batch_positions):
            mel_pert = AudioSignal(
                audio_batch[i:i+1], 
                sample_rate=signal.sample_rate
            ).mel_spectrogram(
                n_mels=n_mels, window_length=hop_length, hop_length=hop_length
            ).squeeze().cpu().numpy()
            
            for j, rel_dist in enumerate(relative_distances):
                mel_frame_idx = int(np.round((pos + rel_dist) * np.prod(encoder_strides) / hop_length))
                if 0 <= mel_frame_idx < mel_recons.shape[-1]:
                    mcd_values[j] += compute_mcd(
                        mel_recons[:, mel_frame_idx:mel_frame_idx+1],
                        mel_pert[:, mel_frame_idx:mel_frame_idx+1]
                    )
    
    mcd_values /= len(positions)
    relative_times = [dist * hop_length / signal.sample_rate for dist in relative_distances]
    
    return np.array(relative_times), mcd_values


def visualize_locality_curve(relative_times, mcd_values, output_path=None):
    """
    Visualize temporal locality curve.
    
    Args:
        relative_times: Array of relative times from perturbation
        mcd_values: Array of MCD values for each time
        output_path: Path to save the figure (if None, returns figure)
    
    Returns:
        matplotlib.figure.Figure object if output_path is None
    """
    fig = plt.figure(figsize=(8, 6))
    plt.plot(relative_times, mcd_values, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Perturbation Time')
    plt.xlabel('Time from Perturbation [s]')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(relative_times), max(relative_times))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Temporal Locality of Perturbation Effects')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='svg' if str(output_path).endswith('.svg') else 'png', dpi=150)
        plt.close()
        return None
    else:
        return fig


def save_evaluation_plots_to_wandb(model, signal, latents, step=0, prefix="eval"):
    """
    Generate and save all evaluation plots to wandb.
    
    Args:
        model: DAC model
        signal: AudioSignal object
        latents: Latent representations
        step: Current training step
        prefix: Prefix for wandb logging
    """
    # Create a unique temporary directory for plots to avoid race conditions
    # when multiple training runs execute on the same machine
    temp_dir_obj = tempfile.mkdtemp(prefix="temp_eval_plots_")
    temp_dir = Path(temp_dir_obj)
    
    try:
        # 1. Latent visualization with PCA
        latent_plot_path = temp_dir / "latent_pca.png"
        visualize_latents_with_pca(signal, latents, n_components=64, 
                                  perform_pca=True, output_path=latent_plot_path)
        wandb.log({f"{prefix}/latent_pca": wandb.Image(str(latent_plot_path))}, step=step)
        
        # 2. Compute condition number and visualize covariance
        cond_number, cov_matrix, _, _ = compute_condition_number(latents)
        cov_plot_path = temp_dir / "covariance.png"
        visualize_covariance_matrix(cov_matrix, cond_number, output_path=cov_plot_path)
        wandb.log({
            f"{prefix}/covariance": wandb.Image(str(cov_plot_path)),
            f"{prefix}/condition_number": cond_number
        }, step=step)
        
        # 3. Smoothness curve
        perturbation_magnitudes = np.logspace(-2, 0, 20)  # Fewer points for speed
        mcd_errors, baseline_mcd = compute_smoothness_curve(
            model, signal, latents, perturbation_magnitudes
        )
        smooth_plot_path = temp_dir / "smoothness.png"
        visualize_smoothness_curve(perturbation_magnitudes, mcd_errors, 
                                 baseline_mcd, output_path=smooth_plot_path)
        wandb.log({
            f"{prefix}/smoothness": wandb.Image(str(smooth_plot_path)),
            f"{prefix}/baseline_mcd": baseline_mcd
        }, step=step)
        
        # 4. Locality curve
        relative_times, mcd_values = compute_locality_curve(
            model, signal, latents, window_before=5, window_after=5  # Smaller window for speed
        )
        locality_plot_path = temp_dir / "locality.png" 
        visualize_locality_curve(relative_times, mcd_values, output_path=locality_plot_path)
        wandb.log({f"{prefix}/locality": wandb.Image(str(locality_plot_path))}, step=step)
        
    finally:
        # Clean up temporary directory and all its contents
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

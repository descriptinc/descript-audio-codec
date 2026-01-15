"""Energy Ratio metric - ratio of reconstructed audio energy to original audio energy."""

import numpy as np


class EnergyRatio:
    """
    Computes the ratio of energy between reconstructed and original audio.
    
    Energy ratio = energy(reconstructed) / energy(original)
    
    A value of 1.0 indicates perfect energy preservation.
    Values > 1.0 indicate energy amplification.
    Values < 1.0 indicate energy reduction.
    """
    
    def __init__(self):
        """Initialize the EnergyRatio metric."""
        pass
    
    def __call__(self, audio_original, audio_reconstructed, sample_rate=None):
        """
        Calculate energy ratio between original and reconstructed audio.
        
        Args:
            audio_original: Original audio signal (numpy array)
            audio_reconstructed: Reconstructed audio signal (numpy array)
            sample_rate: Sample rate (unused, kept for API consistency with other metrics)
        
        Returns:
            float: Energy ratio (reconstructed/original) in dB. Ideal value is 0.0.
        """
        # Calculate energy as sum of squared samples
        energy_original = np.sum(audio_original ** 2) + 1e-10
        energy_reconstructed = np.sum(audio_reconstructed ** 2) + 1e-10
        db = 10 * np.log10(energy_reconstructed / energy_original)
        return db


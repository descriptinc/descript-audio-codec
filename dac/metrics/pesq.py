"""Perceptual Evaluation of Speech Quality (PESQ) metric."""

import torch
import torchaudio
from pesq import pesq as calculate_pesq


class PESQ:
    """
    Perceptual Evaluation of Speech Quality (PESQ) metric class.
    
    PESQ is an ITU-T standard (P.862) for automated assessment of speech quality.
    Returns scores typically between 0 (very bad) and 5 (very good).
    
    Note: This implementation uses wide-band mode. Audio will be automatically 
          resampled to 16kHz if provided at a different sample rate.
    """
    
    def __init__(self):
        """
        Initialize the PESQ metric for wide-band mode (16kHz).
        """
        pass
    
    def __call__(self, audio_original, audio_encoded, sample_rate):
        """
        Calculate PESQ score between original and encoded-decoded audio.
        
        Args:
            audio_original: Original audio signal (numpy array)
            audio_encoded: Encoded-decoded audio signal (numpy array)
            sample_rate: Sample rate of the audio in Hz (will be resampled to 16kHz if needed)
        
        Returns:
            float: PESQ score (typically 0-5, higher is better)
        """
        # PESQ wide-band mode requires 16kHz
        pesq_sample_rate = 16000
        
        # Resample audio if necessary
        if sample_rate != pesq_sample_rate:
            # Convert to tensor, resample, then back to numpy
            resampler = torchaudio.transforms.Resample(sample_rate, pesq_sample_rate)
            
            audio_tensor_original = torch.FloatTensor(audio_original).unsqueeze(0)
            audio_original = resampler(audio_tensor_original).squeeze(0).numpy()
            
            audio_tensor_encoded = torch.FloatTensor(audio_encoded).unsqueeze(0)
            audio_encoded = resampler(audio_tensor_encoded).squeeze(0).numpy()
        
        # Ensure same length
        min_len = min(len(audio_original), len(audio_encoded))
        audio_original = audio_original[:min_len]
        audio_encoded = audio_encoded[:min_len]
        
        # Calculate PESQ
        return calculate_pesq(
            fs=pesq_sample_rate,
            ref=audio_original,
            deg=audio_encoded,
            mode='wb'
        )

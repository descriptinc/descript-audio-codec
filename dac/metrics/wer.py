"""Word Error Rate (WER) metric using Whisper for transcription.

Supports both file-path inputs and in-memory audio arrays to avoid disk I/O
when desired.
"""

import os
from typing import Union

import numpy as np
import torchaudio
import whisper
import jiwer
import torch


class WER:
    """
    Word Error Rate (WER) metric class.
    
    Uses OpenAI's Whisper model to transcribe audio and jiwer to calculate WER.
    WER measures the difference between transcriptions of original and encoded audio.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the WER metric with Whisper turbo model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model = whisper.load_model("turbo", device=device)
    
    def __call__(
        self,
        original: Union[str, os.PathLike, np.ndarray, torch.Tensor],
        encoded: Union[str, os.PathLike, np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> float:
        """
        Calculate WER between transcriptions of original and encoded-decoded audio.

        Args:
            original: Either path to original audio file, or a mono waveform array/tensor
            encoded: Either path to encoded-decoded audio file, or a mono waveform array/tensor
            sample_rate: Sample rate of provided arrays/tensors (ignored for paths)

        Returns:
            float: Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
        """
        def _transcribe_any(x):
            # Accept file path or in-memory audio
            if isinstance(x, (str, os.PathLike)):
                return self.model.transcribe(x, temperature=0.0, best_of=1)["text"]
            # Convert to numpy float32 mono
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().squeeze()
                if x.ndim == 2:
                    x = x.mean(dim=0)
                x = x.float().numpy()
            elif isinstance(x, np.ndarray):
                x = np.asarray(x)
                if x.ndim == 2:
                    x = x.mean(axis=0)
            else:
                raise TypeError("Unsupported input type for WER: expected path, numpy array, or torch tensor")

            # Ensure float32
            x = x.astype(np.float32, copy=False)
            # Resample to 16kHz for Whisper if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                x = resampler(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
            return self.model.transcribe(x, temperature=0.0, best_of=1)["text"]

        with torch.no_grad():
            return jiwer.wer(_transcribe_any(original), _transcribe_any(encoded))
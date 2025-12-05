"""Speaker Identification Metric (SIM) - Simple version with delayed import."""

import speechbrain.inference
import torch
import torchaudio


class SIM:
    """
    Speaker Identification Metric (SIM) class.
    
    Delays SpeechBrain import until first use to avoid initialization conflicts.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the SIM metric."""
        self.device = device
        self._model = speechbrain.inference.EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device},
            )
    
    def __call__(self, audio_original, audio_encoded, sample_rate=16000):
        """
        Calculate SIM score between original and encoded-decoded audio.
        
        Args:
            audio_original: Original audio signal (numpy array)
            audio_encoded: Encoded-decoded audio signal (numpy array)
            sample_rate: Sample rate of the audio (default: 16000)
        
        Returns:
            float: Cosine similarity score between -1 and 1 (higher is better)
        """ 
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_original = resampler(torch.FloatTensor(audio_original).unsqueeze(0)).squeeze(0).numpy()
            audio_encoded = resampler(torch.FloatTensor(audio_encoded).unsqueeze(0)).squeeze(0).numpy()
        
        # Thread-safe computation with cleanup
        with torch.no_grad():
            return torch.nn.functional.cosine_similarity(
                self._model.encode_batch(torch.FloatTensor(
                    audio_original).unsqueeze(0).cpu()).squeeze(1).cpu(),
                self._model.encode_batch(torch.FloatTensor(
                    audio_encoded).unsqueeze(0).cpu()).squeeze(1).cpu()
            ).item()
                
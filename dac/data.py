from typing import Tuple

import numpy as np
from torch import Tensor


def get_reference_audio(
    audio: Tensor,
    reference_length: float = 5.0,
    sample_rate: int = 16000
) -> Tuple[Tensor, Tensor]:
    """

    Parameters
    ----------
    reference_length : float
        Length, in seconds, of reference signal for reference encoder.
    sample_rate : int
        Sample rate of audio data

    Returns
    -------
    reference : Tensor[B x 1 x T]
        Reference audio
    audio : Tensor[B x 1 x T]
        Codec input audio
    """
    ref_side = np.random.choice(["prefix", "suffix"])
    reference_index = int(reference_length * sample_rate)
    if ref_side == "prefix":
        return audio[..., :reference_index], audio[..., reference_index:]
    return audio[..., -reference_index:], audio[..., :-reference_index]

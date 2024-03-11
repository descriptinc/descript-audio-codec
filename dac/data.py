from typing import Tuple

import numpy as np
from torch import Tensor
import torch.nn.functional as F


def get_reference_audio(
    audio: Tensor,
    reference_length: float = 5.0,
    sample_rate: int = 16000,
    ref_side: str = None
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
    if ref_side is None:
        ref_side = np.random.choice(["prefix", "suffix"])
    reference_index = int(reference_length * sample_rate)
    if ref_side == "prefix":
        return audio[..., :reference_index], audio[..., reference_index:], ref_side
    return audio[..., -reference_index:], audio[..., :-reference_index], ref_side


def sample_from_logits(logits, sample=True, temperature=1.0, top_k=None, top_p=None, return_probs: bool = False):
    """Convenience function to sample from a categorial distribution with input as
    unnormalized logits.

    Parameters
    ----------
    logits : Tensor[..., vocab_size]
    sample : bool, optional
        Whether to perform multinomial sampling, by default True
    temperature : float, optional
        Scaling parameter when multinomial samping, by default 1.0
    top_k : int, optional
        Restricts sampling to only `top_k` values acc. to probability,
        by default None
    top_p : float, optional
        Restricts sampling to only those values with cumulative
        probability = `top_p`, by default None

    Returns
    -------
    Tensor[...]
        Sampled tokens
    """
    shp = logits.shape[:-1]

    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[
            ..., :-1
        ]

        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float("inf")

    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )

    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token
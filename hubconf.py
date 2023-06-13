dependencies = [
    "argbind>=0.3.7",
    "audiotools @ git+https://github.com/descriptinc/audiotools.git@0.7.0",
    "einops",
    "numpy",
    "torch",
    "torchaudio",
    "tqdm",
]
from pathlib import Path

import torch

from dac.model import DAC
from dac import __model_version__
from dac import load_model as _load_model


def load_model(pretrained: bool = False, tag: str = __model_version__, **kwargs):
    """Load model

    Parameters
    ----------
    pretrained : bool, optional
        If `True` use the pretrained model weights to initialize
        the model, by default False.
    tag : str, optional
        Model tag, by default 0.0.1
    **kwargs
        Other keyword arguments used to initialize the model.
        These are ignored when `pretrained` is `True`. See
        `dac.model.DAC` for more details.

    Returns
    -------
    dac.model.DAC
        Model object
    """
    if pretrained:
        model_path = (
            Path(torch.hub.get_dir()) / "descript" / tag / "dac" / "weights.pth"
        )
        model = _load_model(tag, load_path=model_path)
    else:
        model = DAC(**kwargs)

    return model

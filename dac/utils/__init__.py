from pathlib import Path

import argbind
from audiotools import ml

import dac

DAC = dac.model.DAC
Accelerator = ml.Accelerator

__MODEL_LATEST_TAGS__ = {
    "44khz": "0.0.1",
    "24khz": "0.0.4",
    "16khz": "0.0.5",
}

__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
}


@argbind.bind(group="download", positional=True, without_prefix=True)
def download(model_type: str = "44khz", tag: str = "latest"):
    """
    Function that downloads the weights file from URL if a local cache is not found.

    Parameters
    ----------
    model_type : str
        The type of model to download. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz".
    tag : str
        The tag of the model to download. Defaults to "latest".

    Returns
    -------
    Path
        Directory path required to load model via audiotools.
    """
    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[model_type]

    download_link = __MODEL_URLS__.get((model_type, tag), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    local_path = (
        Path.home() / ".cache" / "descript" / model_type / tag / "dac" / f"weights.pth"
    )
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


def load_model(
    model_type: str = "44khz",
    tag: str = "latest",
    load_path: str = None,
):
    if not load_path:
        load_path = download(model_type, tag)
    generator, _ = DAC.load(load_path)
    return generator

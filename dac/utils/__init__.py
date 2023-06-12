from pathlib import Path

from audiotools import ml

import dac


DAC = dac.model.DAC
Accelerator = ml.Accelerator


def ensure_default_model(tag: str = dac.__model_version__):
    """
    Function that downloads the weights file from URL if a local cache is not
    found.

    Args:
        tag (str): The tag of the model to download.
    """
    download_link = f"https://github.com/descriptinc/descript-audio-codec/releases/download/{tag}/weights.pth"
    local_path = Path.home() / ".cache" / "descript" / tag / "dac" / f"weights.pth"
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

    # return the path required by audiotools to load the model
    return local_path.parent.parent


def load_model(
    tag: str,
    load_path: str = "",
):
    if not load_path:
        load_path = ensure_default_model(tag)
    kwargs = {
        "folder": load_path,
        "map_location": "cpu",
        "package": False,
    }
    print(f"Loading weights from {kwargs['folder']}")
    generator, _ = DAC.load_from_folder(**kwargs)
    return generator

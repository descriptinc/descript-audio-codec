import torch
import argbind
from pathlib import Path

from audiotools import AudioSignal
from audiotools.data import transforms
from dac.model import DAC
from dac.utils.transforms import PowerNorm
from dac.metrics.eval_utils import visualize_latents_with_pca

@argbind.bind(without_prefix=True)
def visualize(
    audio_file: Path = Path(""),
    model_path: Path = Path(""),
    n_components: int = 64,
    pca: bool = False,
):
    """Visualize mel spectrogram and first n_components PCA components of DAC latents for an audio file.
    
    Args:
        audio_file (Path): Path to input audio file
        model_path (Path): Path to model checkpoint file
        n_components (int): Number of PCA components to visualize
        pca (bool): Whether to perform PCA on latents
    """
    # Load model package
    model = DAC.load(model_path / "best/dac/package.pth")
    model.eval()
    
    # Load and process audio
    signal = AudioSignal(audio_file)
    signal.to(model.device)
    
    # Apply training postprocess via PowerNorm
    _power_norm = PowerNorm(db=-16.0)
    signal = _power_norm._transform(signal)
    
    # Get DAC latents
    with torch.no_grad():
        latents = model.encode(signal.audio_data)
        if isinstance(latents, tuple):
            latents = latents[0]
    
    # Generate and save visualization
    output_path = audio_file.parent / f"latents_{model_path.name}{'_pca' if pca else ''}.svg"
    visualize_latents_with_pca(
        signal, 
        latents, 
        n_components=n_components,
        perform_pca=pca,
        output_path=output_path
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        visualize()
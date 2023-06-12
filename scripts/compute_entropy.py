import argbind
import audiotools as at
import numpy as np
import torch
import tqdm

import dac


@argbind.bind(without_prefix=True, positional=True)
def main(
    folder: str,
    model_path: str,
    n_samples: int = 1024,
    device: str = "cuda",
):
    files = at.util.find_audio(folder)[:n_samples]
    signals = [
        at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=1.0)
        for f in files
    ]

    with torch.no_grad():
        model = dac.model.DAC.load(model_path).to(device)
        model.eval()

        codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(model.device)
            o = model.encode(x.audio_data, x.sample_rate)
            codes.append(o["codes"].cpu())

        codes = torch.cat(codes, dim=-1)
        entropy = []

        for i in range(codes.shape[1]):
            codes_ = codes[0, i, :]
            counts = torch.bincount(codes_)
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))

        pct = sum(entropy) / (10 * len(entropy))
        print(f"Entropy for each codebook: {entropy}")
        print(f"Effective percentage: {pct * 100}%")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()

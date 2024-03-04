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
        (at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=6.0))
        for f in files
    ]
    print(len(signals))

    with torch.no_grad():
        model = dac.model.DAC.load(model_path)
        model.ref_encoder.encoder = torch.load("/data/joseph/dac_encoder.pt")
        model.to(device).eval()

        codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(model.device)
            if x.signal_duration >= 6.0:
                ref_signal, input_signal = dac.get_reference_audio(x.audio_data, 5.0, x.sample_rate)
                signal = at.AudioSignal(input_signal, x.sample_rate).resample(model.sample_rate)
                o = model.encode(signal.audio_data, ref_signal)
                codes.append(o[1].cpu())

        codes = torch.cat(codes, dim=-1)
        print(codes.shape)
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

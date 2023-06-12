import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from encodec import EncodecModel


class Encodec(BaseModel):
    def __init__(self, sample_rate: int = 24000, bandwidth: float = 24.0):
        super().__init__()

        if sample_rate == 24000:
            self.model = EncodecModel.encodec_model_24khz()
        else:
            self.model = EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(bandwidth)
        self.sample_rate = 44100

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 44100,
        n_quantizers: int = None,
    ):
        signal = AudioSignal(audio_data, sample_rate)
        signal.resample(self.model.sample_rate)
        recons = self.model(signal.audio_data)
        recons = AudioSignal(recons, self.model.sample_rate)
        recons.resample(sample_rate)
        return {"audio": recons.audio_data}


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = Encodec()

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]

    print(x.shape, out.shape)

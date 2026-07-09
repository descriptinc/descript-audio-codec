import math
from typing import List

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from einops import rearrange
from einops import repeat
from torch import nn
import torch.nn.functional as F

from .base import CodecMixin
from dac.nn.layers import RMSNorm
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False, use_rmsnorm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=4 if causal else 7, dilation=dilation, causal=causal),
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1, causal=causal),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 16, stride: int = 1, causal: bool = False, dilate: bool = True, use_rmsnorm: bool = True, use_residual: bool = False):
        super().__init__()
        dilation = (2 if causal else 3) if dilate else 1
        self.stride = stride
        self.use_residual = use_residual
        self.output_dim = output_dim
        
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(input_dim, dilation=dilation, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(input_dim, dilation=dilation * dilation, causal=causal, use_rmsnorm=use_rmsnorm),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                causal=causal,
            ),
        )

    def forward(self, x):
        out = self.block(x)
        
        if self.use_residual and self.stride > 1:
            channel_multiplier = self.output_dim // x.shape[1]
            residual = rearrange(x, "b c (t m g) -> b (c m) g t", m=channel_multiplier, g=self.stride // channel_multiplier).mean(dim=2)
            out = out + residual
            
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        multipliers: list = [2, 4, 8, 8],
        d_latent: int = 64,
        causal: bool = False,
        dilate: bool = True,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        power_channel: bool = False,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7
        self.use_residual = use_residual
        self.power_channel = power_channel
        self.d_latent = d_latent
        self.d_model = d_model
        self.strides = strides
        
        # Create first convolution separately
        self.first_conv = WNConv1d(1, d_model, kernel_size=kernel_size, causal=causal)

        # Create EncoderBlocks that increase channels by multipliers as they downsample by strides
        self.block = []
        current_dim = d_model
        for stride, multiplier in zip(strides, multipliers):
            output_dim = current_dim * multiplier
            self.block += [EncoderBlock(current_dim, output_dim, stride=stride, causal=causal, dilate=dilate, use_rmsnorm=use_rmsnorm, use_residual=use_residual)]
            current_dim = output_dim

        # Add RMSNorm and Snake1d to block only when not using residual
        if not use_residual:
            self.block += [
                RMSNorm(current_dim) if use_rmsnorm else nn.Identity(),
                Snake1d(current_dim),
            ]
        
        # Wrap blocks into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = current_dim
        
        # Create final content projection. Power, when enabled, is a side-channel.
        self.final_conv = WNConv1d(current_dim, d_latent, kernel_size=kernel_size, causal=causal)
        self.power_head = WNConv1d(current_dim, 1, kernel_size=kernel_size, causal=causal) if power_channel else None

    def forward(self, x):
        # Apply first convolution
        out = self.first_conv(x)
        
        if self.use_residual:
            # Add residual connection for first conv
            residual = repeat(x, "b c t -> b (c r) t", r=self.d_model // x.shape[1])
            out = out + residual
        
        # Process through main blocks
        features = self.block(out)
        
        # Apply final content projection
        content = self.final_conv(features)
        
        if self.use_residual:
            # Add residual connection for final conv
            residual = rearrange(features, "b (c g) t -> b c g t", c=self.d_latent).mean(dim=2)
            content = content + residual

        if self.power_channel:
            power = self.power_head(features)
            return torch.cat([power, content], dim=1)
        
        return content


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, causal: bool = False, dilate: bool = True, use_rmsnorm: bool = True, use_residual: bool = False):
        super().__init__()
        dilation = (2 if causal else 3) if dilate else 1
        self.stride = stride
        self.use_residual = use_residual
        self.output_dim = output_dim
        
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                causal=causal,
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=dilation, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=dilation * dilation, causal=causal, use_rmsnorm=use_rmsnorm),
        )

    def forward(self, x):
        out = self.block(x)
        
        if self.use_residual and self.stride > 1:
            # DC-AE style residual: parameter-free transformation
            residual = repeat(x, "b c t -> b (c r) t", r=self.stride * self.output_dim // x.shape[1])
            residual = rearrange(residual, "b (c s) t -> b c (t s)", s=self.stride)
                
            out = out + residual
            
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        strides,
        multipliers,
        d_out: int = 1,
        causal: bool = False,
        dilate: bool = True,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
        power_channel: bool = False,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7
        self.use_residual = use_residual
        self.power_channel = power_channel
        self.d_out = d_out
        self.channels = channels

        # Create first conv layer separately
        self.first_conv = WNConv1d(input_channel, channels, kernel_size=kernel_size, causal=causal)
        self.power_scale = nn.Conv1d(1, channels, kernel_size=1) if power_channel else None
        if self.power_scale is not None:
            nn.init.zeros_(self.power_scale.weight)
            nn.init.zeros_(self.power_scale.bias)

        # Add upsampling + MRF blocks
        layers = []
        input_dim = channels
        for stride, multiplier in zip(strides, multipliers):
            output_dim = input_dim // multiplier
            layers += [DecoderBlock(input_dim, output_dim, stride, causal=causal, dilate=dilate, use_rmsnorm=use_rmsnorm, use_residual=use_residual)]
            input_dim = output_dim

        # Add RMSNorm and Snake1d to layers only when not using residual
        if not use_residual:
            layers += [
                RMSNorm(output_dim) if use_rmsnorm else nn.Identity(),
                Snake1d(output_dim),
            ]
        
        # Wrap layers into nn.Sequential
        self.main_layers = nn.Sequential(*layers)
        
        # Always create the final convolution and tanh
        self.final_conv = WNConv1d(output_dim, d_out, kernel_size=kernel_size, causal=causal)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.power_channel:
            power = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            power = None

        # Apply first convolution
        out = self.first_conv(x)
        if power is not None:
            out = out * (1 + self.power_scale(power))
        
        if self.use_residual:
            # Add residual connection for first conv
            residual = repeat(x, "b c t -> b (c r) t", r=self.channels // x.shape[1])
            out = out + residual
        
        # Process through main layers
        features = self.main_layers(out)
        
        # Apply final conv
        out = self.final_conv(features)
        
        if self.use_residual:
            # Add residual connection for final conv
            residual = rearrange(features, "b (c g) t -> b c g t", c=self.d_out).mean(dim=2)
            out = out + residual
        
        # Apply tanh activation
        return self.tanh(out)


class WavLMDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        strides: List[int] = [],
        multipliers: List[int] = [],
        d_out: int = 1024,
        causal: bool = False,
        dilate: bool = True,
        use_rmsnorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7
        self.use_residual = use_residual
        self.d_out = d_out
        self.channels = channels

        # Create first conv layer separately
        self.first_conv = WNConv1d(input_channel, channels, kernel_size=kernel_size, causal=causal)

        # Add upsampling + decoder blocks
        layers = []
        input_dim = channels
        for stride, multiplier in zip(strides, multipliers):
            output_dim = input_dim // multiplier
            layers += [DecoderBlock(input_dim, output_dim, stride, causal=causal, dilate=dilate, use_rmsnorm=use_rmsnorm, use_residual=use_residual)]
            input_dim = output_dim

        self.final_input_dim = input_dim
        
        # Add RMSNorm and Snake1d to layers only when not using residual
        if not use_residual:
            layers += [
                RMSNorm(input_dim) if use_rmsnorm else nn.Identity(),
                Snake1d(input_dim),
            ]
        
        # Wrap layers into nn.Sequential
        self.main_layers = nn.Sequential(*layers)
        
        # Always create the final convolution
        self.final_conv = WNConv1d(input_dim, d_out, kernel_size=kernel_size, causal=causal)

    def forward(self, x):
        # Apply first convolution
        out = self.first_conv(x)
        
        if self.use_residual:
            # Add residual connection for first conv
            residual = repeat(x, "b c t -> b (c r) t", r=self.channels // x.shape[1])
            out = out + residual
        
        # Process through main layers
        features = self.main_layers(out)
        
        # Apply final conv
        out = self.final_conv(features)
        
        if self.use_residual:
            # Add residual connection for final conv
            residual = features
            out = out + residual
        
        return out


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_strides: List[int] = [2, 4, 8, 8],
        encoder_multipliers: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_strides: List[int] = [8, 8, 4, 2],
        decoder_multipliers: List[int] = [8, 8, 4, 2],
        wavlm_decoder_strides: List[int] = [],
        wavlm_decoder_multipliers: List[int] = [],
        latent_noise_max: float = 0.05,  # Maximum standard deviation for noise injection
        sample_rate: int = 44100,
        causal: bool = False,
        dilate: bool = True,
        use_rmsnorm: bool = True,
        use_residual: bool = False,  # DC-AE inspired residual connections
        structured_latent: bool = False,  # Progressive channel dropout for latents
        power_channel: bool = False,  # Use first latent channel for explicit power encoding
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_strides = encoder_strides
        self.encoder_multipliers = encoder_multipliers
        self.decoder_dim = decoder_dim
        self.decoder_strides = decoder_strides
        self.decoder_multipliers = decoder_multipliers
        self.sample_rate = sample_rate
        self.latent_noise_max = latent_noise_max
        self.use_residual = use_residual
        self.structured_latent = structured_latent
        self.power_channel = power_channel

        if latent_dim is None:
            latent_dim = encoder_dim * np.prod(encoder_multipliers)

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_strides)
        self.encoder = Encoder(
            encoder_dim,
            encoder_strides,
            encoder_multipliers,
            latent_dim,
            causal=causal,
            dilate=dilate,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            power_channel=power_channel,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_strides,
            decoder_multipliers,
            causal=causal,
            dilate=dilate,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
            power_channel=power_channel,
        )
        self.wavlm_decoder = WavLMDecoder(
            latent_dim,
            decoder_dim,
            wavlm_decoder_strides,
            wavlm_decoder_multipliers,
            causal=causal,
            dilate=dilate,
            use_rmsnorm=use_rmsnorm,
            use_residual=use_residual,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        if self.power_channel:
            nn.init.zeros_(self.decoder.power_scale.weight)
            nn.init.zeros_(self.decoder.power_scale.bias)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        """Encode given audio data

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        Tensor[B x D x T]
            Encoded latent representation
        """
        return self.encoder(audio_data)

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        training: bool = False,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of input audio, by default None
        training : bool, optional
            Whether in training mode, by default False
            If True, injects random Gaussian noise to the latents with std
            uniformly sampled from [0, latent_noise_max].
            Additionally, if use_residual is True, applies progressive channel 
            dropout (DC-AE 1.5 style) by randomly zeroing out channels after a 
            randomly selected cutoff point to encourage structured latent space.

        Returns
        -------
        dict
            Dictionary containing:
            "z" : Tensor[B x D x T]
                Encoded latent representation (with noise if training=True)
            "z_clean" : Tensor[B x D x T]
                Clean encoded latent representation (without noise)
            "audio" : Tensor[B x 1 x length]
                Reconstructed audio
            "wavlm" : Tensor[B x wavlm_dim x T]
                WavLM embeddings from latent
            "z_augmented" : Tensor[B x D x T]
                Gain-augmented latents when power_channel=True and training=True, 
                otherwise same as z_clean
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z_clean = self.encode(audio_data)
        
        if training:
            # Handle power_channel: encode both original and gain-augmented versions
            if self.power_channel:
                # Sample random gain from -6dB to 6dB
                gain = 10 ** ((torch.rand(audio_data.shape[0], 1, 1, device=audio_data.device) * 12 - 6) / 20)
                audio_augmented = audio_data * gain
                
                # Encode gain-augmented version
                z_augmented = self.encode(audio_augmented)
            else:
                # No augmentation, use clean latents
                z_augmented = z_clean
            
            # Add Gaussian noise with random std between 0 and latent_noise_max
            z = z_clean + torch.randn_like(z_clean) * torch.rand(z_clean.shape[0], 1, 1, device=z_clean.device) * self.latent_noise_max
            
            # Apply progressive channel dropout if structured_latent is enabled (DC-AE 1.5 style)
            if self.structured_latent:
                # Create possible cutoff values
                cutoff_values = torch.arange(z.shape[1] // 8, z.shape[1] + 1, z.shape[1] // 32, device=z.device)
                # Randomly select a cutoff for each batch element
                cutoff_channels = cutoff_values[torch.randint(0, cutoff_values.shape[0], (z.shape[0],), device=z.device)]
                
                # Create a mask that zeros out channels after the cutoff
                channel_mask = (
                    rearrange(torch.arange(z.shape[1], device=z.device), "c -> 1 c 1")
                    <= rearrange(cutoff_channels, "b -> b 1 1")
                ).float()
                
                # Apply the mask to zero out channels after the cutoff
                z = z * channel_mask
                z_clean = z_clean * channel_mask
                z_augmented = z_augmented * channel_mask
        else:
            z = z_clean
            z_augmented = z_clean
            
        x = self.decode(z)
        wavlm_z = z[:, 1:, :] if self.power_channel else z
        return {
            "audio": x[..., :length],
            "z": z,
            "z_clean": z_clean,
            "z_augmented": z_augmented,
            "wavlm": self.wavlm_decoder(wavlm_z),
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

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
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)

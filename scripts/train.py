import os
import sys
import warnings

# Monkey patch torch.load to handle PyTorch 2.6's weights_only default
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, set it to False for compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
import wandb

import dac
from dac.metrics import SIM, PESQ, WER
from dac.metrics.eval_utils import (
    compute_condition_number,
    save_evaluation_plots_to_wandb,
)

from contextlib import contextmanager
from contextlib import nullcontext
warnings.filterwarnings("ignore", category=UserWarning)

class WandbTracker(Tracker):
    """Wrapper around Tracker that adds wandb integration."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_initialized = False
        self.gradient_accumulation_steps = 1  # Default value

    def init_wandb(self, project, name, config):
        """Initialize wandb if not already initialized."""
        if not self.wandb_initialized and self.rank == 0:
            wandb.init(project=project, name=name, config=config)
            self.wandb_initialized = True
            
    def log(self, prefix, reduction="mean", history=True):
        """Override log to also log to wandb."""
        original_log = super().log(prefix, reduction, history)
        
        def wandb_log_fn(fn):
            def wrapper(*args, **kwargs):
                output = fn(*args, **kwargs)
                if self.rank == 0 and self.wandb_initialized and self.step % self.gradient_accumulation_steps == 0:
                    # Convert tensor values to scalars
                    wandb_output = {}
                    for k, v in output.items():
                        if torch.is_tensor(v):
                            wandb_output[k] = v.item()
                        else:
                            wandb_output[k] = v
                    wandb.log(wandb_output, step=self.step // self.gradient_accumulation_steps)
                return output
            return wrapper
        
        return lambda fn: wandb_log_fn(original_log(fn))
        
    def finish(self):
        """Finish wandb run."""
        if self.rank == 0 and self.wandb_initialized:
            wandb.finish()

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(dac.model.DAC)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Import and add our custom PowerNorm transform
from dac.utils.transforms import PowerNorm
# Add PowerNorm to the transforms module so it can be used like other transforms
transforms.PowerNorm = PowerNorm
# Now bind it with the same pattern as other transforms
tfm.PowerNorm = argbind.bind(PowerNorm)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


class EMA:
    """Exponential Moving Average of model parameters and buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: str = None):
        self.decay = float(decay)
        self.device = device
        self.shadow_params = {}
        self.shadow_buffers = {}
        self.backup = {}
        self._register(model)

    @torch.no_grad()
    def _register(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                v = param.detach().clone()
                if self.device is not None:
                    v = v.to(self.device)
                self.shadow_params[name] = v
        for name, buf in model.named_buffers():
            v = buf.detach().clone()
            if self.device is not None:
                v = v.to(self.device)
            self.shadow_buffers[name] = v

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow_params:
                self.shadow_params[name].lerp_(param.detach(), 1.0 - self.decay)
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                self.shadow_buffers[name].lerp_(buf.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.detach().clone()
        for name, buf in model.named_buffers():
            self.backup[f"buffer::{name}"] = buf.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].data)
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                buf.data.copy_(self.shadow_buffers[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        for name, buf in model.named_buffers():
            bname = f"buffer::{name}"
            if bname in self.backup:
                buf.data.copy_(self.backup[bname].data)
        self.backup = {}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state: dict):
        self.decay = float(state.get("decay", self.decay))
        for k, v in state.get("shadow_params", {}).items():
            self.shadow_params[k] = v.to(self.device) if self.device is not None else v
        for k, v in state.get("shadow_buffers", {}).items():
            self.shadow_buffers[k] = v.to(self.device) if self.device is not None else v

@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    wavlm_loss: losses.WavLMLoss
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss
    l2_latents: losses.L2LatentsLoss
    power_contrastive_loss: losses.PowerInvariantContrastiveLoss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker
    ema: EMA = None
    latents_warmup_steps: int = 10000
    
    # Audio quality metrics
    sim_metric: SIM = None
    pesq_metric: PESQ = None
    wer_metric: WER = None

    # Mixed precision bfloat16
    bfloat: bool = False
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)
    l2_latents = losses.L2LatentsLoss()
    wavlm_loss = losses.WavLMLoss(device=accel.device)
    power_contrastive_loss = losses.PowerInvariantContrastiveLoss()
    
    # EMA setup: enabled iff ema_decay > 0
    ema_decay = float(args.get("ema_decay", 0.999))
    ema = None
    if ema_decay > 0.0:
        ema = EMA(accel.unwrap(generator), decay=ema_decay, device=None)

    # Initialize metrics (only on rank 0 to save memory)
    if accel.local_rank == 0:
        sim_metric = None #SIM(device=accel.device)
        pesq_metric = PESQ()
        wer_metric = WER(device=accel.device)
    else:
        sim_metric = pesq_metric = wer_metric = None
    # Load EMA state if resuming
    if resume and ema is not None and "ema.pth" in g_extra:
        try:
            ema.load_state_dict(g_extra["ema.pth"])  
        except Exception:
            pass

    latents_warmup_steps = int(args.get("latents_warmup_steps", 10000))
    # bfloat16 flag picked from config (defaults to False)
    bfloat_flag = bool(args.get("bfloat", False))
    # Gradient accumulation steps (defaults to 1, no accumulation)
    gradient_accumulation_steps = int(args.get("gradient_accumulation_steps", 1))

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        l2_latents=l2_latents,
        wavlm_loss=wavlm_loss,
        power_contrastive_loss=power_contrastive_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        sim_metric=sim_metric,
        pesq_metric=pesq_metric,
        wer_metric=wer_metric,
        ema=ema,
        latents_warmup_steps=latents_warmup_steps,
        bfloat=bfloat_flag,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    if state.ema is not None:
        with state.ema.average_parameters(accel.unwrap(state.generator)):
            out = state.generator(signal.audio_data, signal.sample_rate)
            latents = accel.unwrap(state.generator).encode(signal.audio_data)
    else:
        out = state.generator(signal.audio_data, signal.sample_rate)
        latents = accel.unwrap(state.generator).encode(signal.audio_data)
    
    if isinstance(latents, tuple):
        latents = latents[0]
    
    recons = AudioSignal(out["audio"], signal.sample_rate)
    
    # Compute WavLM losses
    wavlm_cosine_loss, wavlm_mse_loss = state.wavlm_loss(out["wavlm"], signal)
    
    # Compute average condition number for the batch
    condition_numbers = []
    for i in range(latents.shape[0]):
        cond_num, _, _, _ = compute_condition_number(latents[i:i+1])
        condition_numbers.append(cond_num)
    avg_condition_number = sum(condition_numbers) / len(condition_numbers) if condition_numbers else 0
    
    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
        "wavlm/cosine_loss": wavlm_cosine_loss,
        "wavlm/mse_loss": wavlm_mse_loss,
        "condition_number": torch.tensor(avg_condition_number, device=signal.audio_data.device),
    }


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    # When to step (end of accumulation window)
    is_update_step = ((state.tracker.step + 1) % state.gradient_accumulation_steps == 0)

    device_type = accel.device.type if hasattr(accel.device, "type") else ("cuda" if torch.cuda.is_available() else "cpu")
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if state.bfloat else nullcontext()

    # ----------------
    # Forward once
    # ----------------
    with autocast_ctx:
        out = state.generator(signal.audio_data, signal.sample_rate, training=True)
        recons = AudioSignal(out["audio"], signal.sample_rate)
    # downstream losses expect float32
    recons.audio_data = recons.audio_data.float()

    # ----------------
    # D: accumulate discriminator grads ONLY
    # ----------------
    # (3) Ensure D loss does not backprop to G
    recons_for_d = AudioSignal(recons.audio_data.detach(), recons.sample_rate)
    with autocast_ctx:
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons_for_d, signal)

    accel.backward(output["adv/disc_loss"] / state.gradient_accumulation_steps)

    # ----------------
    # G: freeze D, accumulate generator grads ONLY
    # ----------------
    # (1) Freeze D so G loss doesn't write grads into D
    for p in state.discriminator.parameters():
        p.requires_grad_(False)

    with autocast_ctx:
        output["latents/loss"] = state.l2_latents(out["z_clean"])
        gen_loss, feat_loss = state.gan_loss.generator_loss(recons, signal)
        output["adv/gen_loss"] = gen_loss
        output["adv/feat_loss"] = feat_loss

    # float32-safe losses
    output["stft/loss"] = state.stft_loss(recons, signal)
    output["mel/loss"] = state.mel_loss(recons, signal)
    output["waveform/loss"] = state.waveform_loss(recons, signal)

    wavlm_cosine, wavlm_mse = state.wavlm_loss(out["wavlm"], signal)
    output["wavlm/cosine_loss"] = wavlm_cosine
    output["wavlm/mse_loss"] = wavlm_mse

    # Add power contrastive loss if power_channel is enabled
    if accel.unwrap(state.generator).power_channel:
        output["power_contrastive/loss"] = state.power_contrastive_loss(out["z_clean"], out["z_augmented"])

    # latents warmup computed on effective steps (you already did this right)
    curr_lambdas = dict(lambdas)
    if "latents/loss" in curr_lambdas and state.latents_warmup_steps > 0:
        warmup_ratio = min(1.0, max(0.0, state.tracker.step / (state.gradient_accumulation_steps * state.latents_warmup_steps)))
        curr_lambdas["latents/loss"] *= warmup_ratio
        output["other/latents_lambda"] = torch.tensor(curr_lambdas["latents/loss"], device=signal.audio_data.device)

    output["loss"] = sum([v * output[k] for k, v in curr_lambdas.items() if k in output])

    accel.backward(output["loss"] / state.gradient_accumulation_steps)

    # Unfreeze D for the next iteration
    for p in state.discriminator.parameters():
        p.requires_grad_(True)

    # ----------------
    # Step both optimizers together at the end
    # ----------------
    if is_update_step:
        # Guard in case AMP scaler is not present
        accel.scaler.unscale_(state.optimizer_d)
        output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
            state.discriminator.parameters(), 10.0
        )

        accel.scaler.unscale_(state.optimizer_g)
        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
            state.generator.parameters(), 1e3
        )

        # (2) Step both optimizers once per accumulation window
        accel.step(state.optimizer_d)
        accel.step(state.optimizer_g)
        state.scheduler_d.step()
        state.scheduler_g.step()

        state.optimizer_d.zero_grad(set_to_none=True)
        state.optimizer_g.zero_grad(set_to_none=True)

        accel.update()

        if state.ema is not None:
            state.ema.update(accel.unwrap(state.generator))

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in [x * state.gradient_accumulation_steps for x in save_iters]:
        tags.append(f"{state.tracker.step // (1000 * state.gradient_accumulation_steps)}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        # If EMA available, temporarily swap in EMA weights for saving
        if state.ema is not None:
            state.ema.store(accel.unwrap(state.generator))
            state.ema.copy_to(accel.unwrap(state.generator))
            generator_extra["ema.pth"] = state.ema.state_dict()
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        if state.ema is not None:
            state.ema.restore(accel.unwrap(state.generator))
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx):
    state.tracker.print("Saving audio samples to wandb")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    if state.ema is not None:
        with state.ema.average_parameters(accel.unwrap(state.generator)):
            out = state.generator(signal.audio_data, signal.sample_rate)
            # Also get latents for evaluation
            latents = accel.unwrap(state.generator).encode(signal.audio_data)
            if isinstance(latents, tuple):
                latents = latents[0]
    else:
        out = state.generator(signal.audio_data, signal.sample_rate)
        # Also get latents for evaluation
        latents = accel.unwrap(state.generator).encode(signal.audio_data)
        if isinstance(latents, tuple):
            latents = latents[0]
    recons = AudioSignal(out["audio"], signal.sample_rate)

    effective_step = state.tracker.step // state.gradient_accumulation_steps

    # Calculate metrics between original and reconstructed audio
    if state.pesq_metric is not None or state.wer_metric is not None:
        # Lists to store metrics for all samples
        sim_scores = []
        pesq_scores = []
        wer_scores = []
        
        for i in range(signal.batch_size):
            # Get individual signals
            original_audio = signal[i].audio_data.squeeze().cpu().numpy()
            encoded_audio = recons[i].audio_data.squeeze().cpu().numpy()
            
            # Calculate SIM
            if state.sim_metric is not None:
                sim_scores.append(state.sim_metric(original_audio, encoded_audio, signal.sample_rate))
            
            # Calculate PESQ
            if state.pesq_metric is not None:
                pesq_scores.append(state.pesq_metric(original_audio, encoded_audio, signal.sample_rate))
            
            # Calculate WER - save audio to temporary files first
            if state.wer_metric is not None:
                orig_path = f"temp_wer_original_{i}.wav"
                enc_path = f"temp_wer_encoded_{i}.wav"
                signal[i].cpu().write(orig_path)
                AudioSignal(recons[i].audio_data, signal.sample_rate).cpu().write(enc_path)
                
                # Calculate WER using file paths
                wer_scores.append(state.wer_metric(orig_path, enc_path))
                
                # Clean up temporary files
                os.remove(orig_path)
                os.remove(enc_path)
        
        # Log average metrics to wandb
        metrics_to_log = {}
        
        if sim_scores:
            metrics_to_log["metrics/sim"] = sum(sim_scores) / len(sim_scores)
        
        if pesq_scores:
            metrics_to_log["metrics/pesq"] = sum(pesq_scores) / len(pesq_scores)
        
        if wer_scores:
            metrics_to_log["metrics/wer"] = sum(wer_scores) / len(wer_scores)
        
        if metrics_to_log:
            wandb.log(metrics_to_log, step=effective_step)

    # Generate evaluation plots for each sample
    for i in range(signal.batch_size):
        # Get individual signal and latents
        signal_i = signal[i]
        latents_i = latents[:, :, :]  # Keep batch dim for compatibility
        if i < latents.shape[0]:
            latents_i = latents[i:i+1, :, :]
        
        # Save evaluation plots to wandb
        save_evaluation_plots_to_wandb(
            accel.unwrap(state.generator), 
            signal_i, 
            latents_i, 
            step=effective_step,
            prefix=f"eval/sample_{i}"
        )
        
        # Also compute and log individual condition numbers
        cond_number, _, _, _ = compute_condition_number(latents_i)
        wandb.log({f"eval/sample_{i}/condition_number": cond_number}, step=effective_step)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            # Save audio to wandb
            audio_path = f"temp_{k}_sample_{nb}.wav"
            v[nb].cpu().write(audio_path)
            wandb.log({f"audio/{k}/sample_{nb}": wandb.Audio(audio_path)}, step=effective_step)
            os.remove(audio_path)  # Clean up temporary file


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    
    # Initialize tracker with wandb integration
    tracker = WandbTracker(
        log_file=f"{save_path}/log.txt", 
        rank=accel.local_rank,
    )
    
    # Initialize wandb through tracker
    tracker.init_wandb(
        project="descript-audio-codec",
        name=Path(save_path).name,
        config=args
    )

    state = load(args, accel, tracker, save_path)
    
    # Set gradient accumulation steps in tracker for proper wandb logging
    tracker.gradient_accumulation_steps = state.gradient_accumulation_steps
    
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=num_workers > 0,
    )

    # Wrap the functions so that they neatly track in wandb + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track(
            "train",
            num_iters * state.gradient_accumulation_steps,
            completed=state.tracker.step,
        )(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, lambdas)

            last_iter = (tracker.step == num_iters * state.gradient_accumulation_steps - 1) if num_iters is not None else False

            if tracker.step % (sample_freq * state.gradient_accumulation_steps) == 0 or last_iter:
                save_samples(state, val_idx)

            if tracker.step % (valid_freq * state.gradient_accumulation_steps) == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {state.tracker.step // state.gradient_accumulation_steps}")

            if last_iter:
                tracker.finish()
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)

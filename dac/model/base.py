import math
from pathlib import Path
from typing import Union

import torch
import tqdm
from audiotools import AudioSignal


class CodecMixin:
    EXT = ".dac"

    @torch.no_grad()
    def reconstruct(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        overlap_win_duration: float = 5.0,
        overlap_hop_ratio: float = 0.5,
        verbose: bool = False,
        normalize_db: float = -16,
        match_input_db: bool = False,
        mono: bool = False,
        **kwargs,
    ):
        """Reconstructs an audio signal from a file or AudioSignal object.
            This function decomposes the audio signal into overlapping windows
            and reconstructs them one by one. The overlapping windows are then
            overlap-and-added together to form the final output.

        Parameters
        ----------
        audio_path_or_signal : Union[str, Path, AudioSignal]
            audio signal to reconstruct
        overlap_win_duration : float, optional
            overlap window duration in seconds, by default 5.0
        overlap_hop_ratio : float, optional
            overlap hop ratio, by default 0.5
        verbose : bool, optional
            by default False
        normalize_db : float, optional
            normalize db, by default -16
        match_input_db : bool, optional
            set True to match input db, by default False
        mono : bool, optional
            set True to convert to mono, by default False
        Returns
        -------
        AudioSignal
            reconstructed audio signal
        """
        self.eval()
        audio_signal = audio_path_or_signal
        if isinstance(audio_signal, (str, Path)):
            audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

        if mono:
            audio_signal = audio_signal.to_mono()

        audio_signal = audio_signal.clone()
        audio_signal = audio_signal.ffmpeg_resample(self.sample_rate)

        original_length = audio_signal.signal_length
        input_db = audio_signal.ffmpeg_loudness()

        # Fix overlap window so that it's divisible by 4 in # of samples
        sr = audio_signal.sample_rate
        overlap_win_duration = ((overlap_win_duration * sr) // 4) * 4
        overlap_win_duration = overlap_win_duration / sr

        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio()
        overlap_hop_duration = overlap_win_duration * overlap_hop_ratio
        do_overlap_and_add = audio_signal.signal_duration > overlap_win_duration

        nb, nac, nt = audio_signal.audio_data.shape
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)

        if do_overlap_and_add:
            pad_length = (
                math.ceil(audio_signal.signal_duration / overlap_win_duration)
                * overlap_win_duration
            )
            audio_signal.zero_pad_to(int(pad_length * sr))
            audio_signal = audio_signal.collect_windows(
                overlap_win_duration, overlap_hop_duration
            )

        range_fn = range if not verbose else tqdm.trange
        for i in range_fn(audio_signal.batch_size):
            signal_from_batch = AudioSignal(
                audio_signal.audio_data[i, ...], audio_signal.sample_rate
            )
            signal_from_batch.to(self.device)
            _output = self.forward(
                signal_from_batch.audio_data, signal_from_batch.sample_rate, **kwargs
            )

            _output = _output["audio"].detach()
            _output_signal = AudioSignal(_output, self.sample_rate).to(self.device)
            audio_signal.audio_data[i] = _output_signal.audio_data.cpu()

        recons = audio_signal
        recons._loudness = None
        recons.stft_data = None

        if do_overlap_and_add:
            recons = recons.overlap_and_add(overlap_hop_duration)
            recons.audio_data = recons.audio_data.reshape(nb, nac, -1)

        if match_input_db:
            recons.ffmpeg_loudness()
            recons = recons.normalize(input_db)

        recons.truncate_samples(original_length)
        return recons

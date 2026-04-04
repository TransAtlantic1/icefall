#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames


@dataclass
class F5TTSMelConfig:
    target_sample_rate: int = 24000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "F5TTSMelConfig":
        return F5TTSMelConfig(**data)


@register_extractor
class F5TTSMelExtractor(FeatureExtractor):
    name = "f5tts-mel"
    config_type = F5TTSMelConfig

    def __init__(self, config: Optional[F5TTSMelConfig] = None):
        super().__init__(config=config)
        self._mel_stft: Optional[torchaudio.transforms.MelSpectrogram] = None
        self._mel_stft_device: Optional[str] = None

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.target_sample_rate

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def _get_mel_transform(
        self, device: Union[str, torch.device]
    ) -> torchaudio.transforms.MelSpectrogram:
        device_str = str(device)
        if self._mel_stft is None or self._mel_stft_device != device_str:
            self._mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.target_sample_rate,
                n_fft=self.config.n_fft,
                win_length=self.config.win_length,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                power=1,
                center=True,
                normalized=False,
                norm=None,
            ).to(device)
            self._mel_stft_device = device_str
        return self._mel_stft

    def _to_wave_tensor(
        self, samples: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)

        samples = samples.float()
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        elif samples.ndim == 2:
            if samples.shape[0] == 1:
                pass
            elif samples.shape[1] == 1:
                samples = samples.transpose(0, 1)
            else:
                raise ValueError(
                    f"Expected mono audio, but got shape {tuple(samples.shape)}"
                )
        else:
            raise ValueError(f"Unexpected audio tensor shape: {tuple(samples.shape)}")

        return samples

    def _extract_mel(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        device = torch.device(self.device)
        waveform = waveform.to(device)

        if sampling_rate != self.config.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sampling_rate,
                new_freq=self.config.target_sample_rate,
            )

        mel = self._get_mel_transform(device)(waveform)
        mel = mel.clamp(min=1e-5).log()
        return mel

    def _trim_or_pad_features(
        self, features: torch.Tensor, num_samples: int, sampling_rate: int
    ) -> torch.Tensor:
        duration = round(num_samples / sampling_rate, ndigits=12)
        expected_num_frames = compute_num_frames(
            duration=duration,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
        )

        if features.shape[0] > expected_num_frames:
            return features[:expected_num_frames]

        if features.shape[0] < expected_num_frames:
            pad_frames = expected_num_frames - features.shape[0]
            if features.shape[0] == 0:
                return torch.zeros(
                    expected_num_frames,
                    self.config.n_mels,
                    device=features.device,
                    dtype=features.dtype,
                )
            return torch.nn.functional.pad(
                features.unsqueeze(0), (0, 0, 0, pad_frames), mode="replicate"
            ).squeeze(0)

        return features

    def extract(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> np.ndarray:
        waveform = self._to_wave_tensor(samples)
        mel = self._extract_mel(waveform, sampling_rate)
        features = mel.squeeze(0).transpose(0, 1)
        features = self._trim_or_pad_features(
            features=features,
            num_samples=waveform.shape[-1],
            sampling_rate=sampling_rate,
        )
        return features.cpu().numpy()

    def _normalize_batch_input(
        self, samples: Union[torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]
    ) -> List[torch.Tensor]:
        if isinstance(samples, torch.Tensor):
            if samples.ndim == 2:
                return [self._to_wave_tensor(wave).squeeze(0) for wave in samples]
            if samples.ndim == 3:
                return [self._to_wave_tensor(wave).squeeze(0) for wave in samples]
            raise ValueError(f"Unexpected batch tensor shape: {tuple(samples.shape)}")

        return [self._to_wave_tensor(wave).squeeze(0) for wave in samples]

    def extract_batch(
        self,
        samples: Union[torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        sampling_rate: int,
        lengths: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        waveforms = self._normalize_batch_input(samples)
        sample_lengths = list(lengths) if lengths is not None else [
            int(wave.shape[-1]) for wave in waveforms
        ]

        device = torch.device(self.device)
        padded = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0
        ).to(device)

        mel = self._extract_mel(padded, sampling_rate)
        batch_features: List[np.ndarray] = []
        for idx, num_samples in enumerate(sample_lengths):
            feats = mel[idx].transpose(0, 1)
            feats = self._trim_or_pad_features(
                features=feats,
                num_samples=int(num_samples),
                sampling_rate=sampling_rate,
            )
            batch_features.append(feats.cpu().numpy())

        return batch_features

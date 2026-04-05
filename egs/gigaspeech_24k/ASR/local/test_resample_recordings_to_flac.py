#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torchaudio
from lhotse import AudioSource, Recording

from resample_recordings_to_flac import build_cache_path, process_recording


def test_process_recording_resamples_audio_and_rewrites_manifest_entry():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_root = tmp_path / "GigaSpeech"
        cache_root = tmp_path / "cache"
        audio_path = source_root / "audio" / "sample.wav"
        audio_path.parent.mkdir(parents=True)

        waveform = torch.zeros(1, 16000)
        torchaudio.save(str(audio_path), waveform, 16000)

        recording = Recording(
            id="rec-1",
            sources=[AudioSource(type="file", channels=[0], source=str(audio_path))],
            sampling_rate=16000,
            num_samples=16000,
            duration=1.0,
        )

        result = process_recording(
            (
                recording,
                str(source_root),
                str(cache_root),
                24000,
                "flac",
                False,
            )
        )

        assert result[0] == "ok"
        resampled = result[1]
        assert resampled.sampling_rate == 24000
        assert Path(resampled.sources[0].source).suffix == ".flac"
        info = torchaudio.info(str(resampled.sources[0].source))
        assert info.sample_rate == 24000


def test_build_cache_path_uses_manifest_path_without_resolving_symlinks():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_root = tmp_path / "download" / "GigaSpeech"
        source_root.mkdir(parents=True)

        manifest_path = source_root / "audio" / "podcast" / "P0000" / "sample.opus"
        cache_root = tmp_path / "cache"

        cache_path = build_cache_path(
            source_path=str(manifest_path),
            source_root=str(source_root),
            cache_root=str(cache_root),
            target_sample_rate=24000,
            codec="flac",
        )

        assert cache_path == (
            cache_root
            / "24000"
            / "audio"
            / "podcast"
            / "P0000"
            / "sample.flac"
        )


if __name__ == "__main__":
    test_process_recording_resamples_audio_and_rewrites_manifest_entry()
    test_build_cache_path_uses_manifest_path_without_resolving_symlinks()
    print("ok")

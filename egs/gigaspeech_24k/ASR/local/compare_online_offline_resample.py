#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from lhotse import RecordingSet
from lhotse.serialization import load_manifest_lazy_or_eager

_LOCAL_DIR = Path(__file__).resolve().parent
if str(_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIR))

from f5tts_mel_extractor import F5TTSMelConfig, F5TTSMelExtractor


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Compare the old online 24k feature path with the current offline "
            "resample path for a single recording."
        ),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--recording-id", type=str, required=True)
    parser.add_argument(
        "--offline-manifest",
        type=Path,
        default=None,
        help=(
            "Optional resampled-recording manifest. When provided, compare both "
            "the in-memory offline stage-4 output and the saved FLAC that stage 4 wrote."
        ),
    )
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to save the full comparison report as JSON.",
    )
    return parser.parse_args()


def load_recordings(path: Path) -> RecordingSet:
    recordings = load_manifest_lazy_or_eager(path)
    if recordings is None:
        raise ValueError(f"Unable to load manifest from {path}")
    if not isinstance(recordings, RecordingSet):
        raise TypeError(f"Expected RecordingSet in {path}, got {type(recordings)}")
    return recordings


def get_recording(recordings: RecordingSet, recording_id: str):
    try:
        return recordings[recording_id]
    except Exception:
        for recording in recordings:
            if recording.id == recording_id:
                return recording
    raise KeyError(f"Recording {recording_id} not found")


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with 1 or 2 dims, got {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform
    return waveform[:1, :]


def resample_if_needed(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    waveform = ensure_mono(waveform).float()
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(
        waveform,
        orig_freq=orig_sr,
        new_freq=target_sr,
    )


def mse_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    min_len = min(len(a), len(b))
    if min_len == 0:
        raise ValueError("Cannot compare empty arrays.")
    diff = a[:min_len] - b[:min_len]
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "length_a": int(len(a)),
        "length_b": int(len(b)),
        "aligned_length": int(min_len),
        "length_delta": int(len(a) - len(b)),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    return mse_stats(a.squeeze(0).cpu().numpy(), b.squeeze(0).cpu().numpy())


def compare_features(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])
    a2 = a[:rows, :cols]
    b2 = b[:rows, :cols]
    diff = a2 - b2
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "frames_a": int(a.shape[0]),
        "frames_b": int(b.shape[0]),
        "dims_a": int(a.shape[1]),
        "dims_b": int(b.shape[1]),
        "aligned_frames": int(rows),
        "aligned_dims": int(cols),
        "frame_delta": int(a.shape[0] - b.shape[0]),
    }


def format_stats(stats: Dict[str, float]) -> str:
    ordered = []
    for key in (
        "mse",
        "rmse",
        "mae",
        "max_abs",
        "length_delta",
        "frame_delta",
    ):
        if key in stats:
            value = stats[key]
            if isinstance(value, float):
                ordered.append(f"{key}={value:.12g}")
            else:
                ordered.append(f"{key}={value}")
    return ", ".join(ordered)


def main():
    args = get_args()

    source_recordings = load_recordings(args.source_manifest)
    source_recording = get_recording(source_recordings, args.recording_id)

    extractor = F5TTSMelExtractor(
        F5TTSMelConfig(
            target_sample_rate=args.target_sample_rate,
            device=args.device,
        )
    )

    source_path = Path(source_recording.sources[0].source)

    # Old online chain used by Cut/Recording loading + F5 extractor:
    # 1) load with manifest sampling_rate semantics
    # 2) extractor internally resamples to 24k if needed
    online_loaded = torch.from_numpy(source_recording.load_audio())
    online_loaded = ensure_mono(online_loaded)
    online_loaded_sr = source_recording.sampling_rate
    online_resampled = resample_if_needed(
        online_loaded,
        orig_sr=online_loaded_sr,
        target_sr=args.target_sample_rate,
    )
    online_features = extractor.extract(
        online_loaded.squeeze(0),
        sampling_rate=online_loaded_sr,
    )

    # Current stage-4 offline chain:
    # 1) torchaudio.load() raw file directly
    # 2) resample to 24k
    offline_generated_loaded, raw_file_sr = torchaudio.load(str(source_path))
    offline_generated_loaded = ensure_mono(offline_generated_loaded)
    offline_generated_resampled = resample_if_needed(
        offline_generated_loaded,
        orig_sr=raw_file_sr,
        target_sr=args.target_sample_rate,
    )
    offline_generated_features = extractor.extract(
        offline_generated_loaded.squeeze(0),
        sampling_rate=raw_file_sr,
    )

    report = {
        "recording_id": args.recording_id,
        "source_manifest": str(args.source_manifest),
        "offline_manifest": str(args.offline_manifest) if args.offline_manifest else None,
        "source_path": str(source_path),
        "target_sample_rate": args.target_sample_rate,
        "paths": {
            "online_loaded_sample_rate": online_loaded_sr,
            "raw_file_sample_rate": raw_file_sr,
            "online_loaded_num_samples": int(online_loaded.shape[-1]),
            "raw_file_num_samples": int(offline_generated_loaded.shape[-1]),
            "online_resampled_num_samples": int(online_resampled.shape[-1]),
            "offline_generated_num_samples": int(offline_generated_resampled.shape[-1]),
        },
        "waveform": {
            "online_vs_offline_generated": compare_tensors(
                online_resampled,
                offline_generated_resampled,
            ),
        },
        "features": {
            "online_vs_offline_generated": compare_features(
                online_features,
                offline_generated_features,
            ),
        },
    }

    if args.offline_manifest is not None:
        offline_recordings = load_recordings(args.offline_manifest)
        offline_recording = get_recording(offline_recordings, args.recording_id)
        offline_saved_loaded = torch.from_numpy(offline_recording.load_audio())
        offline_saved_loaded = ensure_mono(offline_saved_loaded)
        offline_saved_sr = offline_recording.sampling_rate
        offline_saved_features = extractor.extract(
            offline_saved_loaded.squeeze(0),
            sampling_rate=offline_saved_sr,
        )

        report["paths"]["offline_saved_path"] = str(offline_recording.sources[0].source)
        report["paths"]["offline_saved_sample_rate"] = offline_saved_sr
        report["paths"]["offline_saved_num_samples"] = int(offline_saved_loaded.shape[-1])
        report["waveform"]["online_vs_offline_saved"] = compare_tensors(
            online_resampled,
            offline_saved_loaded,
        )
        report["waveform"]["offline_generated_vs_offline_saved"] = compare_tensors(
            offline_generated_resampled,
            offline_saved_loaded,
        )
        report["features"]["online_vs_offline_saved"] = compare_features(
            online_features,
            offline_saved_features,
        )
        report["features"]["offline_generated_vs_offline_saved"] = compare_features(
            offline_generated_features,
            offline_saved_features,
        )

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"recording_id: {report['recording_id']}")
    print(f"source_path: {report['source_path']}")
    print(
        "sample_rates: "
        f"manifest_online={report['paths']['online_loaded_sample_rate']}, "
        f"raw_file={report['paths']['raw_file_sample_rate']}, "
        f"target={report['target_sample_rate']}"
    )
    print(
        "waveform online_vs_offline_generated: "
        f"{format_stats(report['waveform']['online_vs_offline_generated'])}"
    )
    print(
        "features online_vs_offline_generated: "
        f"{format_stats(report['features']['online_vs_offline_generated'])}"
    )

    if "online_vs_offline_saved" in report["waveform"]:
        print(
            "waveform online_vs_offline_saved: "
            f"{format_stats(report['waveform']['online_vs_offline_saved'])}"
        )
        print(
            "waveform offline_generated_vs_offline_saved: "
            f"{format_stats(report['waveform']['offline_generated_vs_offline_saved'])}"
        )
        print(
            "features online_vs_offline_saved: "
            f"{format_stats(report['features']['online_vs_offline_saved'])}"
        )
        print(
            "features offline_generated_vs_offline_saved: "
            f"{format_stats(report['features']['offline_generated_vs_offline_saved'])}"
        )


if __name__ == "__main__":
    main()

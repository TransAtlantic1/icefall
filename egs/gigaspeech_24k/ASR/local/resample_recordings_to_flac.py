#!/usr/bin/env python3

import argparse
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torchaudio
from lhotse import AudioSource, Recording, RecordingSet
from lhotse.serialization import load_manifest_lazy_or_eager


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--target-sample-rate", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--codec", type=str, default="flac", choices=["flac"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached audio and the output manifest if they already exist.",
    )
    return parser.parse_args()


def load_recordings(path: Path) -> Iterable[Recording]:
    recordings = load_manifest_lazy_or_eager(path)
    if recordings is None:
        raise ValueError(f"Unable to load recordings from {path}")
    return recordings


def build_cache_path(
    source_path: str,
    source_root: str,
    cache_root: str,
    target_sample_rate: int,
    codec: str,
) -> Path:
    source = Path(source_path)
    root = Path(source_root)
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{source} is not under source-root {root}") from exc
    return (Path(cache_root) / str(target_sample_rate) / relative).with_suffix(
        f".{codec}"
    )


def cached_audio_info(path: Path) -> Optional[Tuple[int, int]]:
    if not path.is_file():
        return None
    try:
        info = torchaudio.info(str(path))
    except Exception:
        return None

    if info.sample_rate <= 0 or info.num_frames <= 0:
        return None
    return info.sample_rate, info.num_frames


def build_resampled_recording(
    recording: Recording,
    cached_path: Path,
    target_sample_rate: int,
    num_frames: int,
) -> Recording:
    source = recording.sources[0]
    return Recording(
        id=recording.id,
        sources=[
            AudioSource(
                type="file",
                channels=source.channels,
                source=str(cached_path),
            )
        ],
        sampling_rate=target_sample_rate,
        num_samples=num_frames,
        duration=num_frames / target_sample_rate,
        channel_ids=recording.channel_ids,
        transforms=None,
    )


@lru_cache(maxsize=32)
def get_resampler(orig_freq: int, new_freq: int):
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


def process_recording(job):
    recording, source_root, cache_root, target_sample_rate, codec, overwrite = job
    source = recording.sources[0]
    if source.type != "file":
        return "error", recording.id, "Only file-based recordings are supported."
    if len(recording.sources) != 1:
        return "error", recording.id, "Only single-source recordings are supported."

    source_path = str(source.source)
    target_path = build_cache_path(
        source_path=source_path,
        source_root=source_root,
        cache_root=cache_root,
        target_sample_rate=target_sample_rate,
        codec=codec,
    )

    cached = cached_audio_info(target_path)
    if cached is not None and cached[0] == target_sample_rate and not overwrite:
        return (
            "skipped",
            build_resampled_recording(
                recording=recording,
                cached_path=target_path,
                target_sample_rate=target_sample_rate,
                num_frames=cached[1],
            ),
        )

    try:
        waveform, sample_rate = torchaudio.load(source_path)
        if sample_rate != target_sample_rate:
            waveform = get_resampler(sample_rate, target_sample_rate)(waveform)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(
            f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        torchaudio.save(str(tmp_path), waveform, target_sample_rate, format=codec)
        os.replace(tmp_path, target_path)
        return (
            "ok",
            build_resampled_recording(
                recording=recording,
                cached_path=target_path,
                target_sample_rate=target_sample_rate,
                num_frames=waveform.shape[1],
            ),
        )
    except Exception as exc:
        if target_path.exists():
            cached = cached_audio_info(target_path)
            if cached is not None and cached[0] == target_sample_rate:
                return (
                    "skipped",
                    build_resampled_recording(
                        recording=recording,
                        cached_path=target_path,
                        target_sample_rate=target_sample_rate,
                        num_frames=cached[1],
                    ),
                )
        return "error", recording.id, str(exc)


def handle_result(result, writer):
    status = result[0]
    if status in ("ok", "skipped"):
        writer.write(result[1])
        return status, None

    _, recording_id, error = result
    return status, (recording_id, error)


def main():
    args = get_args()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.output_manifest.is_file() and not args.overwrite:
        logging.info("%s exists - skipping", args.output_manifest)
        return

    recordings = load_recordings(args.input_manifest)

    num_workers = args.num_workers
    total = 0
    written = 0
    skipped = 0
    failed = 0

    jobs = (
        (
            recording,
            str(args.source_root),
            str(args.cache_root),
            args.target_sample_rate,
            args.codec,
            args.overwrite,
        )
        for recording in recordings
    )

    if args.output_manifest.exists():
        args.output_manifest.unlink()

    with RecordingSet.open_writer(args.output_manifest) as writer:
        if num_workers <= 1:
            logging.info(
                "Running resampling in the main process because num_workers=%s",
                num_workers,
            )
            result_iter = map(process_recording, jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            result_iter = executor.map(process_recording, jobs, chunksize=8)

        try:
            for result in result_iter:
                total += 1
                status, error_info = handle_result(result, writer)
                if status in ("ok", "skipped"):
                    written += 1
                    if status == "skipped":
                        skipped += 1
                else:
                    failed += 1
                    recording_id, error = error_info
                    logging.warning("Failed to resample %s: %s", recording_id, error)

                if total % 100 == 0:
                    logging.info(
                        "Processed %s recordings (written=%s, skipped=%s, failed=%s)",
                        total,
                        written,
                        skipped,
                        failed,
                    )
        finally:
            if num_workers > 1:
                executor.shutdown(wait=True)

    logging.info(
        "Finished resampling %s: total=%s, written=%s, skipped=%s, failed=%s",
        args.input_manifest,
        total,
        written,
        skipped,
        failed,
    )
    logging.info("Wrote resampled manifest to %s", args.output_manifest)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

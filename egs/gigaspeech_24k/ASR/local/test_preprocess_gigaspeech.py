#!/usr/bin/env python3

from pathlib import Path
from tempfile import TemporaryDirectory

from lhotse import (
    AudioSource,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.serialization import load_manifest_lazy_or_eager

from preprocess_gigaspeech import preprocess_gigaspeech


def make_recording(recording_id: str, sample_rate: int, source: Path) -> Recording:
    return Recording(
        id=recording_id,
        sources=[
            AudioSource(type="file", channels=[0], source=str(source)),
        ],
        sampling_rate=sample_rate,
        num_samples=sample_rate,
        duration=1.0,
    )


def test_preprocess_uses_resampled_recordings_for_dev_and_sharded_m():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        manifest_dir = tmp_path / "manifests"
        output_dir = tmp_path / "fbank"
        resampled_dir = tmp_path / "manifests_resampled" / "24000"
        manifest_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        resampled_dir.mkdir(parents=True)

        original_audio = tmp_path / "audio" / "original.wav"
        resampled_audio = tmp_path / "audio" / "resampled.flac"
        original_audio.parent.mkdir(parents=True)
        original_audio.touch()
        resampled_audio.touch()

        supervisions = SupervisionSet.from_segments(
            [
                SupervisionSegment(
                    id="sup-dev",
                    recording_id="dev-rec",
                    start=0.0,
                    duration=1.0,
                    text="HELLO <COMMA> WORLD",
                    channel=0,
                ),
                SupervisionSegment(
                    id="sup-m",
                    recording_id="m-rec",
                    start=0.0,
                    duration=1.0,
                    text="KEEP THIS",
                    channel=0,
                ),
                SupervisionSegment(
                    id="sup-test",
                    recording_id="test-rec",
                    start=0.0,
                    duration=1.0,
                    text="TEST <PERIOD> TEXT",
                    channel=0,
                ),
            ]
        )
        SupervisionSet.from_segments([supervisions[0]]).to_file(
            manifest_dir / "gigaspeech_supervisions_DEV.jsonl.gz"
        )
        SupervisionSet.from_segments([supervisions[1]]).to_file(
            manifest_dir / "gigaspeech_supervisions_M.jsonl.gz"
        )
        SupervisionSet.from_segments([supervisions[2]]).to_file(
            manifest_dir / "gigaspeech_supervisions_TEST.jsonl.gz"
        )

        RecordingSet.from_recordings(
            [make_recording("dev-rec", 16000, original_audio)]
        ).to_file(manifest_dir / "gigaspeech_recordings_DEV.jsonl.gz")
        RecordingSet.from_recordings(
            [make_recording("m-rec", 16000, original_audio)]
        ).to_file(manifest_dir / "gigaspeech_recordings_M.jsonl.gz")
        RecordingSet.from_recordings(
            [make_recording("test-rec", 16000, original_audio)]
        ).to_file(
            manifest_dir / "gigaspeech_recordings_TEST.jsonl.gz"
        )

        RecordingSet.from_recordings(
            [make_recording("dev-rec", 24000, resampled_audio)]
        ).to_file(resampled_dir / "gigaspeech_recordings_DEV.jsonl.gz")
        RecordingSet.from_recordings(
            [make_recording("test-rec", 24000, resampled_audio)]
        ).to_file(
            resampled_dir / "gigaspeech_recordings_TEST.jsonl.gz"
        )
        split_dir = resampled_dir / "recordings_M_split_2"
        split_dir.mkdir(parents=True)
        RecordingSet.from_recordings(
            [make_recording("m-rec", 24000, resampled_audio)]
        ).to_file(split_dir / "gigaspeech_recordings_M.00000000.jsonl.gz")

        preprocess_gigaspeech(
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            recordings_manifest_dir=resampled_dir,
        )

        dev_cuts = load_manifest_lazy_or_eager(
            output_dir / "gigaspeech_cuts_DEV_raw.jsonl.gz"
        )
        dev_cut = next(iter(dev_cuts))
        assert dev_cut.recording.sampling_rate == 24000
        assert dev_cut.supervisions[0].text == "HELLO WORLD"

        m_cuts = load_manifest_lazy_or_eager(
            output_dir / "gigaspeech_cuts_M_raw.jsonl.gz"
        )
        m_cut = next(iter(m_cuts))
        assert m_cut.recording.sampling_rate == 24000


if __name__ == "__main__":
    test_preprocess_uses_resampled_recordings_for_dev_and_sharded_m()
    print("ok")

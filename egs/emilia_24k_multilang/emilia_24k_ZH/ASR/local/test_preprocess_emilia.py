#!/usr/bin/env python3

from pathlib import Path

from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

from preprocess_emilia import trim_supervisions_to_recordings_sequentially


def make_recording(recording_id: str, duration: float) -> Recording:
    return Recording(
        id=recording_id,
        sources=[AudioSource(type="file", channels=[0], source=f"{recording_id}.flac")],
        sampling_rate=24000,
        num_samples=int(duration * 24000),
        duration=duration,
    )


def test_trim_supervisions_to_recordings_sequentially_handles_subsequence(tmp_path: Path):
    recordings = RecordingSet.from_recordings(
        [
            make_recording("utt-1", 1.0),
            make_recording("utt-2", 2.0),
            make_recording("utt-3", 3.0),
        ]
    )
    supervisions = SupervisionSet.from_segments(
        [
            SupervisionSegment(
                id="utt-1",
                recording_id="utt-1",
                start=0.0,
                duration=1.0,
                channel=0,
                text="keep",
            ),
            SupervisionSegment(
                id="utt-3",
                recording_id="utt-3",
                start=0.0,
                duration=3.05,
                channel=0,
                text="trim",
            ),
        ]
    )

    output_path = tmp_path / "fixed_supervisions.jsonl.gz"
    stats = trim_supervisions_to_recordings_sequentially(
        recordings=recordings,
        supervisions=supervisions,
        output_path=output_path,
    )
    fixed = list(SupervisionSet.from_file(output_path))

    assert [sup.id for sup in fixed] == ["utt-1", "utt-3"]
    assert fixed[1].duration == 3.0
    assert stats["trimmed_supervisions"] == 1
    assert stats["removed_supervisions"] == 0
    assert stats["skipped_recordings_without_supervision"] == 1


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        test_trim_supervisions_to_recordings_sequentially_handles_subsequence(
            Path(d)
        )
    print("ok")

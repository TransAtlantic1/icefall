#!/usr/bin/env python3

import sys
from pathlib import Path


def require_any(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{directory} missing {pattern}")
    return matches[0]


repo_root = Path(__file__).resolve().parents[3]
root = repo_root.parent / "experiments" / "main_flow_validation" / "giga16k"
required = [
    root / "workspace" / "data" / "fbank" / "gigaspeech_cuts_M.jsonl.gz",
    root / "workspace" / "data" / "fbank" / "gigaspeech_cuts_DEV.jsonl.gz",
    root / "workspace" / "data" / "lang_bpe_500" / "bpe.model",
    root / "exports" / "pretrained.pt",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("Missing required outputs:")
    for item in missing:
        print(item)
    sys.exit(1)

exp_candidates = sorted((root / "exp" / "smoke").glob("zipformer_m_g*"))
if not exp_candidates:
    print("Missing GigaSpeech 16k exp dir under exp/smoke")
    sys.exit(1)

exp_dir = exp_candidates[-1]
checks = [
    exp_dir / "epoch-1.pt",
    exp_dir / "greedy_search",
]
missing = [str(path) for path in checks if not path.exists()]
if missing:
    print("Missing required training/decode artifacts:")
    for item in missing:
        print(item)
    sys.exit(1)

try:
    require_any(exp_dir / "greedy_search", "recogs-*.txt")
    require_any(exp_dir / "greedy_search", "wer-summary-*.txt")
except FileNotFoundError as exc:
    print(exc)
    sys.exit(1)

print(f"giga16k validation files look present in {exp_dir}")

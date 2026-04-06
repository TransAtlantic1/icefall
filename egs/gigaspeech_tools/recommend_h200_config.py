#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def floor_to_step(value: float, step: int) -> int:
    return max(step, int(value // step) * step)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("summary", type=Path, help="Path to smoke_summary.json")
    parser.add_argument(
        "--target-memory-gib",
        type=float,
        default=141.0,
        help="Per-GPU memory on the target training machine.",
    )
    parser.add_argument(
        "--default-usable-memory-gib",
        type=float,
        default=116.0,
        help="Conservative usable memory target on the destination GPU.",
    )
    parser.add_argument(
        "--balanced-usable-memory-gib",
        type=float,
        default=124.0,
        help="More aggressive usable memory target on the destination GPU.",
    )
    parser.add_argument(
        "--default-ceiling",
        type=int,
        default=2000,
        help="Upper cap for the conservative recommended max-duration.",
    )
    parser.add_argument(
        "--balanced-ceiling",
        type=int,
        default=2400,
        help="Upper cap for the aggressive recommended max-duration.",
    )
    parser.add_argument(
        "--round-step",
        type=int,
        default=10,
        help="Round recommendations down to this multiple.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    summary = json.loads(args.summary.read_text())
    peak_reserved = summary["peak_reserved_gib_max"]
    if peak_reserved <= 0:
        raise ValueError("peak_reserved_gib_max must be positive")

    md48 = summary["ranks"][0]["max_duration"]
    default_value = floor_to_step(
        md48 * args.default_usable_memory_gib / peak_reserved,
        args.round_step,
    )
    balanced_value = floor_to_step(
        md48 * args.balanced_usable_memory_gib / peak_reserved,
        args.round_step,
    )

    recommendation = {
        "recipe": summary["recipe"],
        "source_world_size": summary["world_size"],
        "source_max_duration": md48,
        "source_peak_reserved_gib_max": peak_reserved,
        "target_memory_gib": args.target_memory_gib,
        "scaled_default_uncapped": default_value,
        "scaled_balanced_uncapped": balanced_value,
        "production_default": min(default_value, args.default_ceiling),
        "production_balanced": min(balanced_value, args.balanced_ceiling),
    }
    print(json.dumps(recommendation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

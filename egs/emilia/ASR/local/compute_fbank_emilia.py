#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    load_manifest,
    load_manifest_lazy,
)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load_cutset(path: Path) -> CutSet:
    cut_set = CutSet.from_file(path)
    if cut_set is not None:
        return cut_set

    cut_set = load_manifest_lazy(path)
    if cut_set is not None:
        return cut_set

    cut_set = load_manifest(path)
    if cut_set is not None:
        return cut_set

    raise ValueError(f"Unable to load cut set from {path}")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raw-cuts-path", type=Path, required=True)
    parser.add_argument("--output-cuts-path", type=Path, required=True)
    parser.add_argument("--storage-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--batch-duration", type=float, default=600.0)
    return parser.parse_args()


def main():
    args = get_args()
    if args.output_cuts_path.is_file():
        logging.info("%s exists - skipping", args.output_cuts_path)
        return

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info("device: %s", device)

    storage_index = Path(f"{args.storage_path}.lca")
    if storage_index.exists():
        os.remove(storage_index)

    cut_set = load_cutset(args.raw_cuts_path)
    computed = cut_set.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=args.storage_path,
        num_workers=args.num_workers,
        batch_duration=args.batch_duration,
        overwrite=True,
    )
    if computed is not None:
        cut_set = computed

    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False, min_duration=None)
    cut_set.to_file(args.output_cuts_path)
    logging.info("Saved computed cuts to %s", args.output_cuts_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()


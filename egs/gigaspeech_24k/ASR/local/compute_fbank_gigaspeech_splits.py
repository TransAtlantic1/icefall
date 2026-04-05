#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    load_manifest,
    load_manifest_lazy,
)

from f5tts_mel_extractor import F5TTSMelConfig, F5TTSMelExtractor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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

    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of dataloading workers used for reading the audio.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        required=True,
        help="The number of splits of the XL subset",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive).",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Recompute split features even if the cut manifests already exist.",
    )
    return parser.parse_args()


def compute_fbank_gigaspeech_splits(args):
    output_dir = "data/fbank/gigaspeech_M_split"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    raw_paths = sorted(output_dir.glob("gigaspeech_cuts_M_raw.*.jsonl.gz"))
    num_splits = len(raw_paths)
    if args.num_splits != num_splits:
        logging.info(
            "num_splits mismatch: arg=%s, discovered=%s. Using discovered value.",
            args.num_splits,
            num_splits,
        )

    start = args.start
    stop = args.stop
    if stop < start:
        stop = num_splits

    stop = min(stop, num_splits)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = F5TTSMelExtractor(F5TTSMelConfig(device=str(device)))
    logging.info(f"device: {device}")

    for i in range(start, stop):
        raw_cuts_path = raw_paths[i]
        idx = raw_cuts_path.name.replace("gigaspeech_cuts_M_raw.", "").replace(
            ".jsonl.gz", ""
        )
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"gigaspeech_cuts_M.{idx}.jsonl.gz"
        if cuts_path.is_file() and not args.overwrite:
            logging.info(f"{cuts_path} exists - skipping")
            continue
        if cuts_path.is_file():
            logging.info("Removing stale cuts manifest %s", cuts_path)
            cuts_path.unlink()

        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = load_cutset(raw_cuts_path)

        logging.info("Computing features")
        filename = output_dir / f"gigaspeech_feats_M_{idx}.lca"
        if filename.exists():
            logging.info(f"Removing {filename}")
            os.remove(str(filename))

        computed = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/gigaspeech_feats_M_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )
        # Recent lhotse versions may update cuts in place and return None.
        if computed is not None:
            cut_set = computed

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_gigaspeech_splits(args)


if __name__ == "__main__":
    main()

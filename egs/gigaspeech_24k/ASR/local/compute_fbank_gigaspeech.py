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
        default=32,
        help="Number of dataloading workers used for reading the audio.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=1000.0,
        help="The maximum number of audio seconds in a batch.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Recompute features even if the cut manifests already exist.",
    )
    return parser.parse_args()


def compute_fbank_gigaspeech(args):
    in_out_dir = Path("data/fbank")

    subsets = (
        "DEV",
        "TEST",
        "M",
        # "L",
        # "S",
        # "XS",
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = F5TTSMelExtractor(F5TTSMelConfig(device=str(device)))

    logging.info(f"device: {device}")

    for partition in subsets:
        cuts_path = in_out_dir / f"gigaspeech_cuts_{partition}.jsonl.gz"
        if cuts_path.is_file() and not args.overwrite:
            logging.info(f"{cuts_path} exists - skipping")
            continue
        if cuts_path.is_file():
            logging.info("Removing stale cuts manifest %s", cuts_path)
            cuts_path.unlink()

        raw_cuts_path = in_out_dir / f"gigaspeech_cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = load_cutset(raw_cuts_path)

        logging.info("Computing features")

        computed = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{in_out_dir}/gigaspeech_feats_{partition}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )
        # Recent lhotse versions may update cuts in place and return None.
        if computed is not None:
            cut_set = computed
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
    compute_fbank_gigaspeech(args)


if __name__ == "__main__":
    main()

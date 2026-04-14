# Copyright      2021  Piotr Zelasko
# Copyright      2023  Xiaomi Corporation     (Author: Yifan Yang)
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
import inspect
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lhotse
import torch
from lhotse import CutSet, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool

_LOCAL_DIR = Path(__file__).resolve().parent.parent / "local"
if str(_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIR))

from f5tts_mel_extractor import F5TTSMelConfig, F5TTSMelExtractor


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AsrVariableTranscriptDataset(K2SpeechRecognitionDataset):
    def __init__(
        self,
        *args,
        transcript_source: str = "text",
        return_cuts: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transcript_source = transcript_source
        self.return_cuts = True
        self._return_cuts = return_cuts

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        batch = super().__getitem__(cuts)

        if self.transcript_source == "raw_text":
            batch["supervisions"]["text"] = [
                str(
                    ((getattr(supervision, "custom", None) or {}).get("raw_text"))
                    or supervision.text
                    or ""
                )
                for cut in batch["supervisions"]["cut"]
                for supervision in cut.supervisions
            ]

        if not self._return_cuts:
            del batch["supervisions"]["cut"]

        return batch


class EmiliaAsrDataModule:
    """
    Shared DataModule for Emilia single-language ASR experiments.

    It supports one language at a time (`zh` or `en`) and expects manifests
    prepared under the recipe's default artifact-root-aware `fbank/<language>`
    layout unless an explicit language-specific manifest directory is passed.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSets. They control the "
            "effective batch sizes, sampling strategies, and augmentations.",
        )
        group.add_argument(
            "--language",
            type=str,
            default="zh",
            choices=["zh", "en"],
            help="Language-specific Emilia subset to load.",
        )
        group.add_argument(
            "--transcript-source",
            type=str,
            default="raw_text",
            choices=["text", "raw_text"],
            help="Which supervision transcript field to expose as batch text.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=None,
            help=(
                "Path to a directory containing Emilia cut manifests or its parent. "
                "If omitted, use the recipe default under the configured artifact root."
            ),
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a single batch.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, draw batches from buckets of similar duration.",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=15,
            help="The number of buckets for the DynamicBucketingSampler.",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances will be concatenated to minimize padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Maximum duration of a concatenated cut relative to the longest cut.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="Padding amount in seconds inserted between concatenated cuts.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly feature extraction.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled, the examples will be shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop the last batch in training.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will include the originating cuts.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="Number of dataloader workers.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="SpecAugment time warp factor.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="When enabled, mix MUSAN noise into the training data.",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        transforms = []
        if self.args.enable_musan:
            musan_path = self._resolved_manifest_dir() / "musan_cuts.jsonl.gz"
            logging.info("Enable MUSAN: %s", musan_path)
            cuts_musan = load_manifest(musan_path)
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                "Using cut concatenation with duration factor %s and gap %s.",
                self.args.duration_factor,
                self.args.gap,
            )
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info("Time warp factor: %s", self.args.spec_aug_time_warp_factor)
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        train = AsrVariableTranscriptDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            transcript_source=self.args.transcript_source,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            train = AsrVariableTranscriptDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    F5TTSMelExtractor(F5TTSMelConfig())
                ),
                input_transforms=input_transforms,
                transcript_source=self.args.transcript_source,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
                world_size=world_size,
                rank=rank,
            )
        else:
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                world_size=world_size,
                rank=rank,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        return DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        if self.args.on_the_fly_feats:
            validate = AsrVariableTranscriptDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    F5TTSMelExtractor(F5TTSMelConfig())
                ),
                transcript_source=self.args.transcript_source,
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = AsrVariableTranscriptDataset(
                cut_transforms=transforms,
                transcript_source=self.args.transcript_source,
                return_cuts=self.args.return_cuts,
            )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            num_buckets=self.args.num_buckets,
            buffer_size=self.args.num_buckets * 5000,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )

        return DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=min(self.args.num_workers, 2),
            persistent_workers=False,
        )

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        test = AsrVariableTranscriptDataset(
            input_strategy=OnTheFlyFeatures(F5TTSMelExtractor(F5TTSMelConfig()))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            transcript_source=self.args.transcript_source,
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        return DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

    def _resolved_manifest_dir(self) -> Path:
        manifest_dir = Path(self.args.manifest_dir)
        if manifest_dir.name.lower() == self.args.language:
            return manifest_dir
        language_dir = manifest_dir / self.args.language
        if language_dir.exists():
            return language_dir
        return manifest_dir

    def _prefix(self) -> str:
        return f"emilia_{self.args.language}"

    def _cuts_path(self, split: str) -> Path:
        return self._resolved_manifest_dir() / f"{self._prefix()}_cuts_{split}.jsonl.gz"

    def _split_train_cuts(self) -> Optional[CutSet]:
        manifest_dir = self._resolved_manifest_dir()
        split_dirs = sorted(manifest_dir.glob("train_split_*"))
        for split_dir in split_dirs:
            pieces = sorted(split_dir.glob(f"{self._prefix()}_cuts_train.*.jsonl.gz"))
            if pieces:
                logging.info("Loading %s split train manifests in lazy mode", len(pieces))
                return lhotse.combine(load_manifest_lazy(p) for p in pieces)
        return None

    @lru_cache()
    def train_cuts(self) -> CutSet:
        path = self._cuts_path("train")
        if path.is_file():
            logging.info("Loading train cuts from %s", path)
            return load_manifest_lazy(path)

        split_cuts = self._split_train_cuts()
        if split_cuts is not None:
            return split_cuts

        raise FileNotFoundError(f"Could not find Emilia train cuts at {path}")

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        path = self._cuts_path("dev")
        logging.info("Loading dev cuts from %s", path)
        return load_manifest_lazy(path)

    @lru_cache()
    def test_cuts(self) -> CutSet:
        path = self._cuts_path("test")
        logging.info("Loading test cuts from %s", path)
        return load_manifest_lazy(path)


AsrDataModule = EmiliaAsrDataModule

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GigaSpeech ASR recipe under the icefall framework (k2-fsa/icefall). It trains Zipformer-based speech recognition models on the GigaSpeech M subset using **F5-TTS vocos-style mel spectrogram features** (24kHz, 100 mel bins) instead of the standard Kaldi fbank features. A companion baseline experiment at `../gigaspeech_16k/` uses standard 16kHz Kaldi features for comparison.

## Key Commands

### Data Preparation (run from `ASR/` directory)
```bash
# Stage 1: Prepare GigaSpeech manifests (M, DEV, TEST)
bash prepare.sh --stage 1 --stop-stage 1

# Stage 3-6: Preprocess + compute F5-TTS mel features
bash prepare.sh --stage 3 --stop-stage 6

# Stage 8: Prepare BPE vocabulary (vocab_size=500)
bash prepare.sh --stage 8 --stop-stage 8
```

Stages 2 and 7 (MUSAN) are intentionally skipped in this recipe. Stage 0 (download) is skipped because data is softlinked.

### Training
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
cd ASR
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --subset M \
  --enable-musan False \
  --max-duration 700
```

### Decoding
```bash
./zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir zipformer/exp \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

## Environment Dependencies

### Python packages
- `torch` and `torchaudio` for Zipformer training and the custom 24k mel extractor.
- `lhotse` for manifests, feature extraction, and dataloading.
- `k2` for the icefall ASR runtime.
- `sentencepiece` for BPE model loading in train/decode.
- `tensorboard` for `SummaryWriter` logging.
- `wandb` only when running with `--use-wandb True`.
- `lilcom` and `kaldifeat` as part of the standard icefall/lhotse feature pipeline.

### Command-line tools
- `jq` for `prepare.sh --stage 8`.
- `sed`, `find`, `gzip`/`gunzip` for the existing recipe scripts.

### Runtime notes
- The custom extractor mirrors `F5-TTS/src/f5_tts/model/modules.py:get_vocos_mel_spectrogram`, but it does not import the full F5-TTS package at runtime.
- W&B is optional; if enabled, the environment must be authenticated with `wandb login` or `WANDB_API_KEY`.
- For one-dashboard TensorBoard comparison, place 16k and 24k `exp-dir` values under the same parent and run `tensorboard --logdir <parent>`.

## Architecture

### Data Pipeline Flow
```
download/GigaSpeech (16kHz opus)
  → Stage 1: lhotse manifests (data/manifests/)
  → Stage 3: preprocess (OOV filter + text normalize → raw cuts)
  → Stage 4-6: F5TTSMelExtractor (16k→24k resample + vocos mel) → data/fbank/
  → Stage 8: BPE tokenizer → data/lang_bpe_500/
```

### Feature Extraction: F5-TTS vs Standard

The custom extractor in `ASR/local/f5tts_mel_extractor.py` implements a lhotse `FeatureExtractor` that:
1. Resamples 16kHz audio to 24kHz internally via `torchaudio.functional.resample`
2. Computes mel spectrogram identical to `F5-TTS/src/f5_tts/model/modules.py:get_vocos_mel_spectrogram`
3. Parameters: `sr=24000, n_fft=1024, hop=256, n_mels=100, power=1, center=True`
4. Log compression: `mel.clamp(min=1e-5).log()`

### Three Model Directories (only zipformer is primary)
- `zipformer/` — Zipformer2 + RNN-T (primary, feature_dim=100)
- `conformer_ctc/` — Conformer + CTC (secondary)
- `pruned_transducer_stateless2/` — Conformer + pruned RNN-T (secondary)

### Key Abstractions
- **GigaSpeechAsrDataModule** (`zipformer/asr_datamodule.py`): Wraps lhotse CutSet loading, sampling, augmentation (SpecAugment), and DataLoader creation. Supports both precomputed features and on-the-fly extraction.
- **AsrModel** (`zipformer/model.py`): Composes encoder_embed (Conv2dSubsampling) + Zipformer2 encoder + decoder + joiner. Supports combined transducer + CTC loss.
- Feature files use lhotse's `.jsonl.gz` (manifests) and `.lca` (lilcom chunky) formats for lazy loading.

### Companion Baseline: `../gigaspeech_16k/`
- Same code structure, uses standard KaldifeatFbank (16kHz, 80 mel bins, Povey window)
- DEV/TEST features softlinked from junguo's precomputed data
- M features computed from scratch for fair comparison

## Important Paths
- Download data: `ASR/download/` (symlink to shared GigaSpeech corpus)
- Computed features: `ASR/data/fbank/`
- BPE model: `ASR/data/lang_bpe_500/bpe.model`
- Shared utilities: `ASR/shared/` → `icefall/shared/`
- F5-TTS reference: `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/F5-TTS/src/f5_tts/model/modules.py`
- Feature comparison tools: `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/mel_comparison/`

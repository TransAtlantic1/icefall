# Emilia ASR Recipe

This directory contains a single-language ASR recipe for the Emilia dataset:

- `ZH` -> standalone Chinese Zipformer ASR
- `EN` -> standalone English Zipformer ASR

The implementation lives in [ASR](./ASR).

## What Is Included

- Shared data prep entrypoint: [ASR/prepare.sh](./ASR/prepare.sh)
- Emilia-specific prep scripts: [ASR/local](./ASR/local)
- Shared non-streaming Zipformer train/decode/export: [ASR/zipformer](./ASR/zipformer)

## Environment

Install a working `icefall` environment first. At minimum, this recipe expects:

- `torch`
- `torchaudio`
- `k2`
- `lhotse`
- `kaldifeat`
- `sentencepiece`
- `tensorboard`

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install torch/torchaudio for your CUDA or CPU setup first.
# Example only:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install k2 -f https://k2-fsa.github.io/k2/cuda.html
pip install "lhotse[kaldi,orjson]" kaldifeat
pip install -r requirements.txt
pip install -e .
```

System tools that are commonly needed:

```bash
ffmpeg
sox
```

## Expected Dataset Layout

This recipe assumes Emilia is organized like:

```text
download/Emilia/
  ZH/
    *.jsonl
    ...
  EN/
    *.jsonl
    ...
```

Each JSONL entry should contain the fields used by [ASR/local/prepare_emilia.py](./ASR/local/prepare_emilia.py):

- `id`
- `wav`
- `text`
- `duration`
- `speaker` (recommended; falls back to `id`)
- `language` (optional if the file already belongs to `ZH/` or `EN/`)
- `dnsmos` (optional; preserved as metadata)

`wav` is expected to be a path relative to the language subdirectory, e.g. relative to `download/Emilia/ZH` or `download/Emilia/EN`.

## Quick Start

Change into the recipe directory:

```bash
cd egs/emilia/ASR
```

Prepare Chinese:

```bash
./prepare.sh \
  --language zh \
  --dataset-root /path/to/Emilia
```

Prepare English:

```bash
./prepare.sh \
  --language en \
  --dataset-root /path/to/Emilia
```

Outputs are separated by language:

- manifests: `data/manifests/zh`, `data/manifests/en`
- features: `data/fbank/zh`, `data/fbank/en`
- BPE/lang: `data/lang_bpe_zh_2000`, `data/lang_bpe_en_500`
- experiments: `zipformer/exp-zh`, `zipformer/exp-en`

## Data Prep Stages

[ASR/prepare.sh](./ASR/prepare.sh) runs:

1. Build split manifests from Emilia JSONL and audio paths
2. Normalize text and create raw cuts
3. Compute dev/test fbank
4. Split train raw cuts
5. Compute train fbank split by split
6. Combine processed train cuts
7. Prepare MUSAN features
8. Train BPE and build the lang directory

Useful knobs:

- `--stage` / `--stop-stage`: run only part of the pipeline
- `--max-jsonl-files`: small smoke test on a few JSONL shards
- `--max-utterances`: small smoke test on a subset of utterances
- `--num-splits`: number of train splits for feature extraction
- `--start` / `--stop`: run only a slice of train split jobs
- `--speed-perturb true|false`: enable or disable train-time prep perturbation

Example smoke test:

```bash
./prepare.sh \
  --language zh \
  --dataset-root /path/to/Emilia \
  --stop-stage 2 \
  --max-jsonl-files 1 \
  --max-utterances 200
```

## Training

Chinese:

```bash
./zipformer/train.py \
  --language zh \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 600
```

English:

```bash
./zipformer/train.py \
  --language en \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 600
```

Defaults are language-aware:

- `ZH`
  - manifests from `data/fbank/zh`
  - BPE from `data/lang_bpe_zh_2000`
  - exp dir `zipformer/exp-zh`
- `EN`
  - manifests from `data/fbank/en`
  - BPE from `data/lang_bpe_en_500`
  - exp dir `zipformer/exp-en`

You can still override `--manifest-dir`, `--bpe-model`, `--lang-dir`, or `--exp-dir` manually.

## Decode

Chinese:

```bash
./zipformer/decode.py \
  --language zh \
  --epoch 30 \
  --avg 9 \
  --decoding-method greedy_search \
  --max-duration 600
```

English:

```bash
./zipformer/decode.py \
  --language en \
  --epoch 30 \
  --avg 9 \
  --decoding-method greedy_search \
  --max-duration 600
```

Scoring is language-specific inside the recipe:

- `ZH`: normalized character-oriented scoring
- `EN`: normalized word-oriented scoring

## Export

Chinese:

```bash
./zipformer/export.py \
  --language zh \
  --epoch 30 \
  --avg 9 \
  --jit 1
```

English:

```bash
./zipformer/export.py \
  --language en \
  --epoch 30 \
  --avg 9 \
  --jit 1
```

## Text Normalization Rules

Normalization is implemented in [ASR/local/text_normalization.py](./ASR/local/text_normalization.py).

- `ZH`
  - normalize full-width/half-width forms
  - lowercase embedded English
  - remove visible punctuation
  - collapse whitespace
  - keep digits unchanged
- `EN`
  - lowercase
  - convert punctuation to spaces
  - collapse whitespace
  - keep digits unchanged

## Notes

- This recipe currently supports only `zh` and `en`.
- It trains two separate single-language models, not a multilingual shared model.
- Dataset files and generated artifacts should stay out of git. Sync code with GitHub; keep `download/`, `data/`, and `zipformer/exp-*` local to each machine.

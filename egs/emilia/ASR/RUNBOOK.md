# Emilia ASR RUNBOOK

This runbook matches commit `9fce537a`.

Canonical pipeline: `prepare.sh`.

Feature setup in this recipe:
- Audio target: 32 kHz
- Offline audio cache: enabled by default
- Acoustic feature: Kaldi-style fbank
- Feature dim for training: 80

`prepare_data.sh` still exists, but the maintained end-to-end chain is `prepare.sh`. Use `prepare_data.sh` only as a legacy convenience wrapper for a single-machine fast path.

## 1. Environment

Run everything from:

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia/ASR
```

Recommended environment:

```bash
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/opt/conda/envs/icefall/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/envs/icefall/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Default dataset root in code is `./download/Emilia`. In practice you will usually override it explicitly:

```bash
DATASET_ROOT=/inspire/dataset/emilia/fc71e07
LANG=zh
```

## 2. Stage map

`prepare.sh` stages:

1. Stage 0: build Lhotse manifests
2. Stage 1: split train recordings into recording shards
3. Stage 2: optional MUSAN manifest prep
4. Stage 3: offline resample recordings to 32 kHz FLAC cache
5. Stage 4: normalize text and build raw cuts
6. Stage 5: compute dev/test features
7. Stage 6: split train raw cuts
8. Stage 7: compute train features by split
9. Stage 8: optional MUSAN features
10. Stage 9: combine split train cut manifests
11. Stage 10: prepare BPE language dir

Important defaults:
- `target_sample_rate=32000`
- `use_resampled_audio=true`
- `speed_perturb=true`
- `enable_musan=false`
- `recording_num_splits=1000`
- `num_splits=100`
- `num_workers=20`
- `batch_duration=600`

Training-side MUSAN default in `zipformer/asr_datamodule.py` is `true`, so if you keep data prep default `enable_musan=false`, you must also pass `--enable-musan false` to training.

## 3. Canonical full data prep

Full chain:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 0 \
  --stop-stage 10
```

Recommended split execution:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 0 \
  --stop-stage 4
```

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 5 \
  --stop-stage 10 \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

Why `--feature-num-workers 0`: this avoids Docker `/dev/shm` issues during batched feature extraction.

## 4. Multi-instance offline resampling

Stage 3 is CPU-oriented and is safe to shard across multiple CPU instances as long as each instance gets a disjoint shard range.

Example with two CPU instances:

Instance A:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 0 \
  --stop-stage 1
```

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 3 \
  --stop-stage 3 \
  --resample-start 0 \
  --resample-stop 500 \
  --resample-num-workers 32
```

Instance B:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 3 \
  --stop-stage 3 \
  --resample-start 500 \
  --resample-stop 1000 \
  --resample-num-workers 32
```

Then continue on one machine:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 4 \
  --stop-stage 10 \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

## 5. Multi-GPU feature extraction

Feature extraction in stages 5 and 7 uses GPU automatically when CUDA is visible.

One-machine single-process version:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 5 \
  --stop-stage 7 \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

Manual shard parallelism for stage 7 across multiple GPUs or machines:

Worker 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 7 \
  --stop-stage 7 \
  --feature-start 0 \
  --feature-stop 25 \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

Worker 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --stage 7 \
  --stop-stage 7 \
  --feature-start 25 \
  --feature-stop 50 \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

Adjust the ranges to cover all `train_split_<N>` shards.

## 6. Optional MUSAN path

If you want MUSAN:

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --enable-musan true \
  --stage 2 \
  --stop-stage 2
```

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --enable-musan true \
  --stage 8 \
  --stop-stage 8
```

If you do not prepare MUSAN, train with `--enable-musan false`.

## 7. Expected artifacts

Core outputs:
- `data/manifests/<lang>/emilia_<lang>_recordings_{train,dev,test}.jsonl.gz`
- `data/manifests_resampled/<lang>/32000/...`
- `data/fbank/<lang>/emilia_<lang>_cuts_{dev,test,train}.jsonl.gz`
- `data/fbank/<lang>/train_split_<N>/emilia_<lang>_cuts_train.*.jsonl.gz`
- `data/lang_bpe_<lang>_<vocab_size>/`

Training can read either:
- combined train cuts `data/fbank/<lang>/emilia_<lang>_cuts_train.jsonl.gz`
- or split train cuts under `data/fbank/<lang>/train_split_*`

## 8. Training

Default experiment dir:
- zh: `zipformer/exp-zh`
- en: `zipformer/exp-en`

Recommended 8-GPU training with TensorBoard and W&B in one shared project:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

```bash
python zipformer/train.py \
  --world-size 8 \
  --language "$LANG" \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 1000 \
  --enable-musan false \
  --tensorboard true \
  --use-wandb true \
  --wandb-project emilia-asr \
  --wandb-group "${LANG}-compare" \
  --wandb-run-name "emilia-${LANG}-32k" \
  --wandb-tags "emilia,32k,kaldifeat"
```

TensorBoard logs are written to:

```text
zipformer/exp-<lang>/tensorboard
```

Resume training:

```bash
python zipformer/train.py \
  --world-size 8 \
  --language "$LANG" \
  --start-epoch 11 \
  --use-fp16 1 \
  --max-duration 1000 \
  --enable-musan false \
  --use-wandb true \
  --wandb-project emilia-asr \
  --wandb-group "${LANG}-compare" \
  --wandb-run-name "emilia-${LANG}-32k"
```

## 9. Decode

Example decode:

```bash
python zipformer/decode.py \
  --language "$LANG" \
  --epoch 30 \
  --avg 15 \
  --exp-dir "zipformer/exp-${LANG}" \
  --max-duration 600 \
  --decoding-method greedy_search
```

## 10. Quick verification

Check feature manifests:

```bash
python - <<'PY'
from lhotse import load_manifest_lazy
for split in ["dev", "test", "train"]:
    p = f"data/fbank/zh/emilia_zh_cuts_{split}.jsonl.gz"
    try:
        cuts = load_manifest_lazy(p)
        first = next(iter(cuts))
        print(split, first.num_features)
    except Exception as e:
        print(split, e)
PY
```

Check training CLI:

```bash
python zipformer/train.py --help | rg 'wandb|tensorboard|language|exp-dir'
```

Check decode CLI:

```bash
python zipformer/decode.py --help >/tmp/emilia_decode_help.txt
```

## 11. Notes

- Stage 3 offline resampling writes FLAC cache files and updated recording manifests. Later feature extraction reads the cached audio through Lhotse, not through manual WAV conversion.
- Stage 9 is recommended but not strictly required for training because the datamodule can fall back to split train manifests.
- If you intentionally skip offline cache, pass `--use_resampled_audio false`. In this recipe that means keeping the original 32 kHz path.

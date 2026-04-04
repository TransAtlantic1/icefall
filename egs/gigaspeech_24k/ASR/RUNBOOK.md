# GigaSpeech 24k Runbook

This runbook describes the full execution flow for the 24k F5-TTS mel experiment.

Data preparation is assumed to run on a CPU-only instance.
Training and decoding must run on a GPU instance.

## 1. Environment Setup

Open a new shell and run:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall:${PYTHONPATH}
```

If you want W&B tracking, use a shared project and group:

```bash
export WANDB_PROJECT=gigaspeech-f5tts-vs-fbank
export WANDB_GROUP=gsm-compare-20260403
```

Create a shared experiment root for TensorBoard comparison:

```bash
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gsm_compare_20260403
mkdir -p ${EXP_ROOT}
```

## 2. Go To The Recipe Directory

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR
```

## 3. Optional Sanity Check

On a CPU-only instance, do not run `train.py` or `decode.py --help`, because the
GPU-enabled `k2` package will try to load CUDA driver libraries. Run those checks
only on the GPU training instance.

## 4. Prepare GigaSpeech Manifests

Generate manifests for `M`, `DEV`, and `TEST`:

```bash
bash prepare.sh --stage 1 --stop-stage 1 --cpu-only true
```

## 5. Preprocess And Compute 24k Features

This step uses `local/f5tts_mel_extractor.py`. The extractor resamples 16kHz input to 24kHz internally and produces 100-dim vocos-style log-mel features.

Recommended explicit execution:

```bash
python local/preprocess_gigaspeech.py --cpu-only true
touch data/fbank/.preprocess_complete
python local/compute_fbank_gigaspeech.py --num-workers 32 --batch-duration 1000
```

Equivalent stage-based execution:

```bash
bash prepare.sh --stage 3 --stop-stage 4 --cpu-only true
```

## 6. Prepare BPE

Run stage 8:

```bash
bash prepare.sh --stage 8 --stop-stage 8 --cpu-only true
```

## 7. Verify Feature Dimension

Check that the computed features are 100-dimensional:

```bash
python - <<'PY'
from lhotse import load_manifest_lazy
cuts = load_manifest_lazy("data/fbank/gigaspeech_cuts_DEV.jsonl.gz")
first = next(iter(cuts))
print("num_features =", first.num_features)
PY
```

Expected output:

```text
num_features = 100
```

## 8. Launch Training

Switch to a GPU training instance before running this step.

Train on 8 GPUs with TensorBoard and W&B enabled:

```bash
python zipformer/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --use-fp16 1 \
  --subset M \
  --enable-musan False \
  --max-duration 700 \
  --tensorboard True \
  --use-wandb True \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-group ${WANDB_GROUP} \
  --wandb-run-name gsm-m-24k-f5tts \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,24k,f5tts-mel \
  --exp-dir ${EXP_ROOT}/24k
```

## 9. Launch Decoding

Run decoding on the GPU training instance after training completes.

After training completes, decode with checkpoint averaging and write the best DEV/TEST WER back to the same W&B run:

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir ${EXP_ROOT}/24k \
  --max-duration 600 \
  --decoding-method modified_beam_search \
  --use-wandb True \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-group ${WANDB_GROUP} \
  --wandb-run-name gsm-m-24k-f5tts \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,24k,f5tts-mel
```

## 10. Check Outputs

Important outputs:

- `${EXP_ROOT}/24k/tensorboard/`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-dev-*.txt`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-test-*.txt`

## 11. Notes

- `stage 5-6` are not part of the main `M` experiment flow.
- The training script defaults `--enable-musan` to `False`.
- On CPU-only data prep instances, use `--cpu-only true` for `prepare.sh --stage 1/3/8` and `local/preprocess_gigaspeech.py`.
- For a quick smoke test before the full run, reduce epochs and batch duration first.

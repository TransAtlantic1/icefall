# GigaSpeech 16k Runbook

This runbook describes the full execution flow for the 16k Kaldi fbank baseline experiment.

Data preparation is assumed to run on a CPU-only instance.
Training and decoding must run on a GPU instance.

## 1. Environment Setup

Open a new shell and run:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall:${PYTHONPATH}
```

If you want W&B tracking, use the same project and group as the 24k experiment:

```bash
export WANDB_PROJECT=gigaspeech-f5tts-vs-fbank
export WANDB_GROUP=gsm-compare-20260403
```

Use the same shared experiment root as the 24k experiment:

```bash
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gsm_compare_20260403
mkdir -p ${EXP_ROOT}
```

## 2. Go To The Recipe Directory

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR
```

## 3. Prepare Shared DEV/TEST Features And BPE

If you already have a populated `egs/gigaspeech/ASR` workspace, you can reuse the
shared DEV/TEST features and `lang_bpe_500`. Otherwise, skip this section and
generate the assets locally via `prepare.sh`.

```bash
export SOURCE_GIGASPEECH_ASR=/path/to/icefall/egs/gigaspeech/ASR

mkdir -p data/fbank

ln -sf ${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_DEV.jsonl.gz data/fbank/
ln -sf ${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_DEV.lca data/fbank/

ln -sf ${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_TEST.jsonl.gz data/fbank/
ln -sf ${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_TEST.lca data/fbank/

ln -sf ${SOURCE_GIGASPEECH_ASR}/data/lang_bpe_500 data/lang_bpe_500
```

## 4. Optional Sanity Check

On a CPU-only instance, do not run `train.py` or `decode.py --help`, because the
GPU-enabled `k2` package will try to load CUDA driver libraries. Run those checks
only on the GPU training instance.

## 5. Prepare GigaSpeech Manifests

Generate manifests for `M`, `DEV`, and `TEST`:

```bash
bash prepare.sh --stage 1 --stop-stage 1 --cpu-only true
```

## 6. Preprocess And Compute 16k Features

This recipe keeps the standard Kaldifeat fbank extractor. Since DEV and TEST may
already be linked in `data/fbank/`, the actual work is mainly for `M`.

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

## 7. Verify Feature Dimension

Check that the DEV features are 80-dimensional:

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
num_features = 80
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
  --wandb-run-name gsm-m-16k-fbank \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,16k,fbank-baseline \
  --exp-dir ${EXP_ROOT}/16k
```

## 9. Launch Decoding

Run decoding on the GPU training instance after training completes.

After training completes, decode with checkpoint averaging and write the best DEV/TEST WER back to the same W&B run:

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir ${EXP_ROOT}/16k \
  --max-duration 600 \
  --decoding-method modified_beam_search \
  --use-wandb True \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-group ${WANDB_GROUP} \
  --wandb-run-name gsm-m-16k-fbank \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,16k,fbank-baseline
```

## 10. Check Outputs

Important outputs:

- `${EXP_ROOT}/16k/tensorboard/`
- `${EXP_ROOT}/16k/modified_beam_search/wer-summary-dev-*.txt`
- `${EXP_ROOT}/16k/modified_beam_search/wer-summary-test-*.txt`

## 11. Notes

- `stage 5-6` are not part of the main `M` experiment flow.
- The training script defaults `--enable-musan` to `False`.
- On CPU-only data prep instances, use `--cpu-only true` for `prepare.sh --stage 1/3` and `local/preprocess_gigaspeech.py`.
- This recipe is a 16 kHz metadata/W&B variant of the standard GigaSpeech
  Kaldifeat fbank baseline. It does not introduce a separate offline-resampling
  stage in this directory.
- The baseline can reuse shared DEV/TEST features and `lang_bpe_500`, but
  computes `M` locally for a fair comparison.

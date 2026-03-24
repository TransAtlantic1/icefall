# GigaSpeech M WER Plan (No Existing Code Changes)

This directory implements the experiment workflow you requested:

1. Baseline lock
2. Core ablations
3. Decode sweep
4. Error analysis
5. Unified result collection

Only new helper scripts are added here. Existing training/decoding code is untouched.

## Prerequisites

- Run from an environment where `torch`, `k2`, `lhotse`, and `sentencepiece` are installed.
- Use a machine with NVIDIA GPUs for practical training speed.
- Ensure GigaSpeech M data/manifests and BPE model are ready under `egs/gigaspeech/ASR`.

## Directory Layout

- `run_baseline.sh`: baseline training + decode (`greedy_search`, `modified_beam_search`)
- `run_ablations.sh`: augmentation/loss/batch-duration ablations
- `run_decode_sweep.sh`: decode-only hyperparameter sweep (`avg`, `use-averaged-model`, `beam-size`)
- `collect_results.py`: aggregate WER + timing + training metrics
- `error_analysis.py`: aggregate substitution/deletion/insertion stats and heuristic categories
- `run_full_plan.sh`: one-command wrapper for the entire workflow
- `results/`: generated CSV/Markdown/JSON outputs

## Quick Start

```bash
cd egs/gigaspeech/ASR/experiments/m_wer_plan

# 1) Baseline (two seeds by default)
./run_baseline.sh \
  --gpus 0,1 \
  --seeds 42,777 \
  --num-epochs 40 \
  --max-duration 1000 \
  --bpe-model data/lang_bpe_500/bpe.model

# 2) Core ablations
./run_ablations.sh \
  --gpus 0,1 \
  --seeds 42,777 \
  --stage all \
  --num-epochs 20 \
  --bpe-model data/lang_bpe_500/bpe.model

# 3) Decode sweep on a trained checkpoint
./run_decode_sweep.sh \
  --gpus 0 \
  --exp-dir zipformer/exp_m_baseline_seed42 \
  --epoch 40 \
  --avg-list 5,9,15 \
  --beam-size-list 2,4,6,8 \
  --use-averaged-model-list 0,1 \
  --bpe-model data/lang_bpe_500/bpe.model

# 4) Collect summary tables
python3 collect_results.py --asr-dir ../../

# 5) Error analysis over all errs files
python3 error_analysis.py \
  --asr-dir ../../ \
  --pattern "zipformer/exp_m_*/**/errs-*.txt"
```

## Notes

- Scripts always use `--subset M`.
- Result logs are appended to CSV in `results/`.
- Error category heuristics are text-only approximations:
  - `digit`: token contains any digit
  - `proper_noun_like`: token is TitleCase (best-effort)
  - `liaison_like`: token contains `'` or `-`


#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS="0,1"
SEEDS="42,777"
BASELINE_EPOCHS=40
ABLATION_EPOCHS=20
BPE_MODEL="data/lang_bpe_500/bpe.model"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --asr-dir PATH                 ASR recipe dir (default: ${ASR_DIR})
  --gpus IDS                     GPU ids for training (default: ${GPUS})
  --seeds CSV                    Seeds list (default: ${SEEDS})
  --baseline-epochs N            Baseline epochs (default: ${BASELINE_EPOCHS})
  --ablation-epochs N            Ablation epochs (default: ${ABLATION_EPOCHS})
  --bpe-model PATH               BPE model relative to ASR dir (default: ${BPE_MODEL})
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr-dir) ASR_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --baseline-epochs) BASELINE_EPOCHS="$2"; shift 2 ;;
    --ablation-epochs) ABLATION_EPOCHS="$2"; shift 2 ;;
    --bpe-model) BPE_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

cd "${SCRIPT_DIR}"

echo "[1/5] Baseline stage"
./run_baseline.sh \
  --asr-dir "${ASR_DIR}" \
  --gpus "${GPUS}" \
  --seeds "${SEEDS}" \
  --num-epochs "${BASELINE_EPOCHS}" \
  --max-duration 1000 \
  --decode-max-duration 1000 \
  --decode-avg 9 \
  --bpe-model "${BPE_MODEL}"

echo "[2/5] Ablation stage"
./run_ablations.sh \
  --asr-dir "${ASR_DIR}" \
  --gpus "${GPUS}" \
  --seeds "${SEEDS}" \
  --stage all \
  --num-epochs "${ABLATION_EPOCHS}" \
  --decode-avg 9 \
  --decode-max-duration 1000 \
  --bpe-model "${BPE_MODEL}" \
  --decode-methods modified_beam_search

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
DECODE_GPU="${GPUS%%,*}"
for seed in "${SEED_ARR[@]}"; do
  echo "[3/5] Decode sweep for baseline seed ${seed}"
  ./run_decode_sweep.sh \
    --asr-dir "${ASR_DIR}" \
    --gpus "${DECODE_GPU}" \
    --exp-dir "zipformer/exp_m_baseline_seed${seed}" \
    --epoch "${BASELINE_EPOCHS}" \
    --avg-list 5,9,15 \
    --use-averaged-model-list 0,1 \
    --beam-size-list 2,4,6,8 \
    --decode-max-duration 1000 \
    --bpe-model "${BPE_MODEL}"
done

echo "[4/5] Collect summary tables"
python3 collect_results.py --asr-dir "${ASR_DIR}"

echo "[5/5] Error analysis"
python3 error_analysis.py \
  --asr-dir "${ASR_DIR}" \
  --pattern "zipformer/exp_m_*/**/errs-*.txt"

echo "Full plan execution finished."


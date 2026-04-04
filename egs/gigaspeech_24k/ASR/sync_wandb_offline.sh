#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
icefall_root="$(cd -- "${script_dir}/../../.." && pwd)"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]] && ! command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "icefall" ]]; then
  conda activate icefall
fi

exp_root_default="$(cd -- "${icefall_root}/.." && pwd)/experiments/gigaspeech_24k_train"
exp_root="${EXP_ROOT:-${exp_root_default}}"
wandb_dir="${1:-${exp_root}/wandb_offline}"

if [[ ! -d "${wandb_dir}" ]]; then
  echo "Offline W&B directory does not exist: ${wandb_dir}" >&2
  exit 1
fi

exec wandb sync --include-offline "${wandb_dir}"

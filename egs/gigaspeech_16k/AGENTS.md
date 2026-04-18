# Recipe Notes

## Current Mainline

This directory currently uses the shortened 16 kHz mainline under `ASR/`.

Mainline chain:
1. `ASR/prepare.sh`
2. `ASR/run_train_offline.sh`
3. `ASR/zipformer/decode.py`
4. `ASR/zipformer/export.py`

Mainline docs should stay short and only describe the stable path above.

## Recent Cleanup Process

The recent cleanup for this recipe followed this sequence:
1. Review uncommitted changes first and avoid touching unrelated edits.
2. Back up the long historical runbooks, results, and smoke helpers.
3. Simplify `ASR/README.md`, `ASR/RUNBOOK.md`, and `ASR/RESULTS.md` so they only describe the current mainline.
4. Move validation scripts out of `ASR/` and into repo-level test recipes.
5. Keep all minimal real-data validation inputs and outputs under `experiments/`, not under recipe `data/`.

## Where Old Intermediate Ops Scripts Went

Use these locations:
- Current minimal validation scripts: `test/recipes/giga16k/`
- Current watcher helpers: `asr_op/giagspeech/watcher/`
- Archived historical docs and removed helper scripts: `asr_op/giagspeech/backup/`

If you are searching for an older script name:
- `run_smoke_train_offline.sh`: archived in `asr_op/giagspeech/backup/`, replaced by `test/recipes/giga16k/run_smoke_train.sh`
- `auto_decode_checkpoints.sh`: active copy is now `asr_op/giagspeech/watcher/auto_decode_checkpoints.sh`
- older long runbook/results text: archived in `asr_op/giagspeech/backup/`

## Current Validation Chain

The validated experiment-only chain is:
1. `test/recipes/giga16k/prepare_minimal_real_data.sh`
2. `test/recipes/giga16k/run_smoke_train.sh`
3. `test/recipes/giga16k/run_decode_export.sh`
4. `test/recipes/giga16k/validate_outputs.py`

Validation outputs must stay under:
- `../experiments/main_flow_validation/giga16k/`

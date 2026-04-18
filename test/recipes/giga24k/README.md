# GigaSpeech 24k Validation

最小真实数据验证顺序：

1. `prepare_minimal_real_data.sh`
2. `run_smoke_train.sh`
3. `run_decode_export.sh`
4. `validate_outputs.py`

默认所有输入和产物都写到 `../experiments/main_flow_validation/giga24k/`。

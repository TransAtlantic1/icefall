# Emilia 24k ASR 运行手册

标准主流程脚本：`prepare.sh`。

当前主线配置：

- 目标采样率：`24 kHz`
- 特征：F5-TTS 风格 log-mel
- 输入维度：`100`
- 默认主流程：`stage 0 -> 10`

采样率原则：

- 直接从原始 source 一次重采样到 `24 kHz`
- 不走中间采样率级联
- stage 4 在生成 raw cuts 前会修正 `supervision.end > recording.duration` 的条目

## 1. 环境

所有命令在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATASET_ROOT=/inspire/dataset/emilia/fc71e07
export ARTIFACT_ROOT=/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public/emilia/fc71e07/icefall_emilia_zh_24k
```

## 2. Stage 对照

- `stage 0`：生成 manifests
- `stage 1`：切分 train recordings
- `stage 3`：离线重采样到 `24 kHz`
- `stage 4`：文本归一化并生成 raw cuts
- `stage 5`：生成 `dev/test` 特征
- `stage 6`：切分 train raw cuts
- `stage 7`：生成 train split 特征
- `stage 9`：合并 train cut manifests
- `stage 10`：准备 `lang_hybrid_zh`

默认参数：

- `recording_num_splits=1000`
- `feature_num_splits=100`
- `feature_num_workers=20`
- `feature_batch_duration=1000`
- `feature_device=auto`

## 3. 数据准备

完整主路径：

```bash
bash prepare.sh \
  --language zh \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 0 \
  --stop-stage 10
```

如果只想做隔离的最小真实数据验证，不要把产物写回主链路目录，直接使用：

```bash
bash /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/emilia24k/prepare_minimal_real_data.sh
```

这个脚本会把所有产物写到：

```text
/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/main_flow_validation/emilia24k/
```

## 4. 训练

配置文件：`configs/train_zh.yaml`

手工训练示例：

```bash
python zipformer/train.py \
  --config configs/train_zh.yaml \
  --language zh \
  --artifact-root "$ARTIFACT_ROOT" \
  --world-size 8 \
  --master-port 12460
```

最小 smoke：

```bash
bash /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/emilia24k/run_smoke_train.sh \
  --mode smoke
```

## 5. 解码

```bash
python zipformer/decode.py \
  --language zh \
  --artifact-root "$ARTIFACT_ROOT" \
  --epoch 10 \
  --avg 1 \
  --exp-dir "$ARTIFACT_ROOT/exp/zipformer/emilia-zh-24k-h200-md1000" \
  --bpe-model "$ARTIFACT_ROOT/data/lang_hybrid_zh/bpe.model" \
  --lang-dir "$ARTIFACT_ROOT/data/lang_hybrid_zh" \
  --decoding-method greedy_search
```

## 6. 导出

```bash
python zipformer/export.py \
  --language zh \
  --artifact-root "$ARTIFACT_ROOT" \
  --epoch 10 \
  --avg 1 \
  --exp-dir "$ARTIFACT_ROOT/exp/zipformer/emilia-zh-24k-h200-md1000" \
  --tokens "$ARTIFACT_ROOT/data/lang_hybrid_zh/tokens.txt"
```

## 7. 补充

- 最小真实数据验证脚本在 [test/recipes/emilia24k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/emilia24k)
- 可选 watcher 在 [asr_op/emilia/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/emilia/watcher)
- 历史长篇机器分片、现场路径和旧 smoke 文本已移出主线文档，原文保存在 `asr_op/emilia/backup/`


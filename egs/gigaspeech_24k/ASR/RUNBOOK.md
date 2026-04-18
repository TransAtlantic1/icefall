# GigaSpeech 24k ASR 运行手册

标准主流程脚本：`prepare.sh`。

当前主线配置：

- 目标采样率：`24 kHz`
- 特征：`100` 维 F5-TTS 风格 log-mel
- 默认主流程：`stage 1 -> 3 -> 4 -> 5 -> 6`
- `stage 7-8` 是可选的 split 特征链路，不属于默认训练主线

采样率原则：

- 直接从原始 source 一次重采样到 `24 kHz`
- 不走 `raw -> 16 kHz -> 24 kHz` 的级联重采样

## 1. 环境

所有命令在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## 2. Stage 对照

- `stage 1`：生成 `M/DEV/TEST` manifests
- `stage 3`：切分 `M` recordings
- `stage 4`：离线重采样到 `24 kHz`
- `stage 5`：文本预处理并生成 raw cuts
- `stage 6`：生成 `DEV/TEST/M` 主特征
- `stage 10`：准备 `lang_bpe_500`

默认参数：

- `stage=1`
- `stop_stage=6`
- `recording_num_splits=100`
- `resample_num_workers=32`
- `feature_num_workers=32`
- `feature_batch_duration=1000`

## 3. 数据准备

标准主路径：

```bash
bash prepare.sh \
  --stage 1 \
  --stop-stage 6 \
  --cpu-only true \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

如果要把 stage 4 拆成分片：

```bash
bash run_resample_shard.sh --instance-index 0
```

如果 `data/lang_bpe_500` 还不存在，再补一次：

```bash
bash prepare.sh \
  --stage 10 \
  --stop-stage 10
```

## 4. 训练

推荐入口：

```bash
bash run_train_offline.sh
```

当前 launcher 默认：

- `NUM_EPOCHS=40`
- `MAX_DURATION=1000`
- `WORLD_SIZE=4`
- `WANDB_MODE=offline`

## 5. 解码

```bash
python zipformer/decode.py \
  --epoch 40 \
  --avg 9 \
  --exp-dir /path/to/exp \
  --manifest-dir data/fbank \
  --bpe-model data/lang_bpe_500/bpe.model \
  --lang-dir data/lang_bpe_500 \
  --max-duration 1000 \
  --decoding-method greedy_search
```

## 6. 导出

```bash
python zipformer/export.py \
  --epoch 40 \
  --avg 9 \
  --exp-dir /path/to/exp \
  --tokens data/lang_bpe_500/tokens.txt
```

## 7. 补充

- 最小真实数据验证脚本在 [test/recipes/giga24k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/giga24k)
- 可选 watcher 在 [asr_op/giagspeech/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/giagspeech/watcher)
- 历史长篇 smoke / H200 / W&B 现场记录已移出主线文档，原文保存在 `asr_op/giagspeech/backup/`

# GigaSpeech 16k ASR 运行手册

标准主流程脚本：`prepare.sh`。

当前主线配置：

- 音频采样率：`16 kHz`
- 特征：`80` 维 Kaldifeat fbank
- 默认主流程：`stage 1 -> 3 -> 4`
- `stage 5-6` 是可选的 split 特征链路，不属于默认训练主线

## 1. 环境

所有命令在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## 2. Stage 对照

- `stage 1`：生成 `M/DEV/TEST` manifests
- `stage 3`：文本预处理并生成 raw cuts
- `stage 4`：计算 `DEV/TEST/M` 特征
- `stage 8`：准备 `lang_bpe_500`

默认参数：

- `stage=1`
- `stop_stage=4`
- `stage4_num_workers=32`
- `stage4_batch_duration=1000`

当前稳定的 `stage 4` 参数：

```bash
bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 0 \
  --stage4-batch-duration 500
```

## 3. 数据准备

标准主路径：

```bash
bash prepare.sh \
  --stage 1 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 0 \
  --stage4-batch-duration 500
```

如果 `data/lang_bpe_500` 还不存在，再补一次：

```bash
bash prepare.sh \
  --stage 8 \
  --stop-stage 8
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

如需覆盖：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=12374 \
NUM_EPOCHS=1 \
MAX_DURATION=300 \
bash run_train_offline.sh
```

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

- 最小真实数据验证脚本在 [test/recipes/giga16k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/giga16k)
- 可选 watcher 在 [asr_op/giagspeech/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/giagspeech/watcher)
- 历史长篇 smoke / H200 / W&B 现场记录已移出主线文档，原文保存在 `asr_op/giagspeech/backup/`

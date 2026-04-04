# GigaSpeech 16k 运行手册

本文档说明 `egs/gigaspeech_16k/ASR` 这个 recipe 的推荐执行流程。

这个 recipe 是标准的 Kaldifeat fbank 流水线，加上 16 kHz 实验元信息和可选的 W&B 记录。
它不在本目录中额外引入单独的离线重采样阶段。

数据准备可以在 CPU-only 机器上完成。
训练和解码建议在 GPU 机器上完成。

## 1. 进入目录

从 Icefall 仓库根目录进入：

```bash
cd egs/gigaspeech_16k/ASR
```

## 2. 环境准备

激活你平时用于 Icefall 的环境。如果你的本地环境需要，也请确保仓库根目录已经加入 `PYTHONPATH`。

如果要做统一跟踪，可以设置这些可选环境变量：

```bash
export WANDB_PROJECT=gigaspeech-compare
export WANDB_GROUP=gigaspeech-16k-vs-24k
export EXP_ROOT=/path/to/experiments/gigaspeech_compare
mkdir -p "${EXP_ROOT}"
```

如果不使用 W&B，后面的命令里去掉 `--use-wandb` 相关参数即可。

## 3. 可选：复用共享的 DEV/TEST 特征和 BPE

如果你已经有一个准备好的 `egs/gigaspeech/ASR` 工作区，可以直接复用其中的 DEV/TEST 特征和 `lang_bpe_500`：

```bash
export SOURCE_GIGASPEECH_ASR=/path/to/icefall/egs/gigaspeech/ASR

mkdir -p data/fbank

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_DEV.jsonl.gz" data/fbank/
ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_DEV.lca" data/fbank/

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_TEST.jsonl.gz" data/fbank/
ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_TEST.lca" data/fbank/

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/lang_bpe_500" data/lang_bpe_500
```

如果这些资源还没有准备好，可以跳过这一步，后续在本地生成。

## 4. 准备 manifests

生成 `M`、`DEV` 和 `TEST` 的 manifests：

```bash
bash prepare.sh --stage 1 --stop-stage 1 --cpu-only true
```

## 5. 预处理并计算特征

该 recipe 保持标准的 Kaldifeat fbank 提取方式。
如果第 3 步已经复用了 DEV 和 TEST 的特征，那么这一步实际主要是在处理子集 `M`。

如果你是在共享实例上做这一步，建议也一起确认宿主机内存、`/dev/shm` 和 cgroup 上限：

```bash
free -h
df -h /dev/shm
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

本 recipe 已验证成功的一组 `M` 子集配置是：

- `--stage4-num-workers 0`
- `--stage4-batch-duration 500`

这次成功运行后的关键产物是：

- `data/fbank/gigaspeech_feats_M.lca` 约 `111G`
- `data/fbank/gigaspeech_cuts_M.jsonl.gz` 约 `2.1G`

这组参数适合：

- 单独跑 `16k stage 4`
- 宿主机和 cgroup 仍有明显内存余量
- 你更关心稳定完成 `M`，而不是把音频读取并发开得很高

参数设计原则：

- 对 `16k` 来说，`batch-duration 500` 已经验证可用，不必保守回退到更小值。
- `num-workers 0` 是这次成功经验的重要部分；它减少了 DataLoader 预取和共享内存波动。
- 如果未来要提速，先尝试把 `num-workers` 提到 `2`，而不是继续把 `batch-duration` 提高到 `1000`。
- 不建议让 `16k stage 4` 和 `24k stage 4` 在同一个受限 cgroup 中并跑。

推荐显式执行：

```bash
python local/preprocess_gigaspeech.py --cpu-only true
touch data/fbank/.preprocess_complete
python local/compute_fbank_gigaspeech.py \
  --num-workers 0 \
  --batch-duration 500
```

等价的 stage 方式：

```bash
bash prepare.sh \
  --stage 3 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 0 \
  --stage4-batch-duration 500
```

## 6. 检查特征维度

确认预计算特征是 80 维：

```bash
python - <<'PY'
from lhotse import load_manifest_lazy

cuts = load_manifest_lazy("data/fbank/gigaspeech_cuts_DEV.jsonl.gz")
first = next(iter(cuts))
print("num_features =", first.num_features)
PY
```

预期输出：

```text
num_features = 80
```

## 7. 训练 Zipformer

多卡训练示例：

```bash
python zipformer/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --use-fp16 1 \
  --subset M \
  --enable-musan False \
  --max-duration 700 \
  --tensorboard True \
  --exp-dir "${EXP_ROOT}/16k"
```

带 W&B 的版本：

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
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name gsm-m-16k-fbank \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,16k,fbank-baseline \
  --exp-dir "${EXP_ROOT}/16k"
```

## 8. 解码

解码示例：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/16k" \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

带 W&B 的版本：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/16k" \
  --max-duration 600 \
  --decoding-method modified_beam_search \
  --use-wandb True \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name gsm-m-16k-fbank \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,16k,fbank-baseline
```

## 9. 预期输出

重要输出包括：

- `${EXP_ROOT}/16k/tensorboard/`
- `${EXP_ROOT}/16k/modified_beam_search/wer-summary-dev-*.txt`
- `${EXP_ROOT}/16k/modified_beam_search/wer-summary-test-*.txt`
- `${EXP_ROOT}/16k/wandb_run_id.txt`，当启用 W&B 时会生成

## 10. 备注

- `stage 5-6` 是可选的 split-based 特征计算辅助流程，不属于主 `M` 训练路径。
- 训练 datamodule 默认把 `--enable-musan` 设为 `False`。
- 在 CPU-only 机器上做预处理时，请使用 `--cpu-only true`。
- 如果你的本地 `k2` 构建依赖 CUDA 库，避免在 CPU-only 机器上执行 `train.py` 或 `decode.py --help`。

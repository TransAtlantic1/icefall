# GigaSpeech 24k 运行手册

本文档说明 `egs/gigaspeech_24k/ASR` 这个 recipe 的推荐执行流程。

这个 recipe 的核心目标是在 GigaSpeech ASR 流水线中使用 24 kHz F5-TTS 风格 log-mel 特征：

- 原始音频主要仍然是 16 kHz
- 在特征提取阶段离线重采样到 24 kHz
- 输出 100 维特征
- 训练、解码和导出入口统一消费这套 100 维特征

数据准备可以在 CPU-only 机器上完成。
训练和解码建议在 GPU 机器上完成。

## 1. 进入目录

从 Icefall 仓库根目录进入：

```bash
cd egs/gigaspeech_24k/ASR
```

## 2. 环境准备

激活你平时用于 Icefall 的环境。如果你的本地环境需要，也请确保仓库根目录已经加入 `PYTHONPATH`。

例如：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall:${PYTHONPATH}
```

如果要做统一跟踪，可以设置这些可选环境变量：

```bash
export WANDB_PROJECT=gigaspeech-f5tts-vs-fbank
export WANDB_GROUP=gsm-compare-20260403
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gsm_compare_20260403
mkdir -p "${EXP_ROOT}"
```

如果不使用 W&B，后面的命令里去掉 `--use-wandb` 相关参数即可。

## 3. 可选：环境自检

如果当前机器是 CPU-only 机器，不要运行 `train.py` 或 `decode.py --help`，因为 GPU 版本的 `k2` 可能会在导入时尝试加载 CUDA 相关库。

这类检查建议只在真正的 GPU 训练机上进行。

## 4. 准备 manifests

生成 `M`、`DEV` 和 `TEST` 的 manifests：

```bash
bash prepare.sh --stage 1 --stop-stage 1 --cpu-only true
```

## 5. 预处理并计算 24k 特征

这一步使用 `local/f5tts_mel_extractor.py`。
提取器会在内部完成 16 kHz -> 24 kHz 的重采样，并输出 100 维 log-mel 特征。

在共享实例上跑这一步之前，先确认你真正能用到的内存上限，而不只是宿主机总内存：

```bash
free -h
df -h /dev/shm
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

判读建议：

- `free -h` 用来确认宿主机总内存和当前空闲内存。
- `df -h /dev/shm` 用来确认共享内存大小；只有在 `--num-workers > 0` 时它才更重要。
- `memory.max` 才是当前容器或作业真正的 cgroup 内存上限；如果它明显小于宿主机总内存，OOM kill 会先按这个上限触发。
- `memory.events` 里如果已经出现过 `oom_kill`，说明当前会话组曾经被 cgroup OOM kill 过。

结合本 recipe 的实际运行经验：

- `gigaspeech_16k` 曾经在 `--stage4-num-workers 0 --stage4-batch-duration 500` 下成功跑完 `M`。
- 同样的 `500` 对 `gigaspeech_24k` 不够稳，因为这里会在线做 `16 kHz -> 24 kHz` 重采样，且输出是 `100` 维特征，单 batch 的瞬时内存和累计主存压力都更大。
- 在共享实例上不要让 `24k stage 4` 和 `16k stage 4` 并跑。

如果目标实例确认有大约 `500 GiB` 内存和 `500 GiB` `/dev/shm`，推荐把 `24k` 的 `M` 子集特征提取放到该实例上单独跑，并使用下面的参数分层策略：

- 保守首跑：`--stage4-num-workers 2 --stage4-batch-duration 150`
- 推荐默认：`--stage4-num-workers 4 --stage4-batch-duration 200`
- 机器稳定、监控确认内存余量仍很大后，再尝试：`--stage4-num-workers 4 --stage4-batch-duration 250`
- 不建议一开始就回到 `--stage4-batch-duration 500`

参数设计原则：

- `batch-duration` 决定每批总音频秒数，是影响峰值内存的主开关，优先调它。
- `num-workers` 只影响音频读取和 DataLoader 预取，不会改变主提特征逻辑；它带来的收益通常小于 `batch-duration` 的风险。
- 如果首要目标是稳定，优先把 `num-workers` 控制在 `2-4`。
- 如果当前机器是 CPU-only 机器，建议显式清空 `CUDA_VISIBLE_DEVICES`，避免错误地落到某张卡上。

推荐显式执行：

```bash
python local/preprocess_gigaspeech.py --cpu-only true
touch data/fbank/.preprocess_complete
CUDA_VISIBLE_DEVICES='' python local/compute_fbank_gigaspeech.py \
  --num-workers 4 \
  --batch-duration 200
```

等价的 stage 方式：

```bash
CUDA_VISIBLE_DEVICES='' bash prepare.sh \
  --stage 3 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 4 \
  --stage4-batch-duration 200
```

如果你要对 `M` 子集做拆分并行计算，也可以使用 `stage 5-6`，但这不是主 `M` 训练路径的必需步骤。

如果你希望先用更保守的参数冒烟，再切换到推荐默认值，可以用：

```bash
CUDA_VISIBLE_DEVICES='' nohup bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 2 \
  --stage4-batch-duration 150 \
  > stage4_24k_cpu.log 2>&1 &
```

如果上面的保守配置在 `20-30` 分钟内稳定推进，且 `memory.current` 远低于 `memory.max`，再换成推荐默认值：

```bash
CUDA_VISIBLE_DEVICES='' nohup bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 4 \
  --stage4-batch-duration 200 \
  > stage4_24k_cpu.log 2>&1 &
```

监控建议：

```bash
tail -f stage4_24k_cpu.log
watch -n 10 free -h
watch -n 10 'cat /sys/fs/cgroup/memory.current; echo; cat /sys/fs/cgroup/memory.events'
```

## 6. 准备 BPE

运行 stage 8：

```bash
bash prepare.sh --stage 8 --stop-stage 8 --cpu-only true
```

## 7. 检查特征维度

确认预计算特征是 100 维：

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
num_features = 100
```

## 8. 训练 Zipformer

切换到 GPU 训练机后再执行这一步。

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
  --exp-dir "${EXP_ROOT}/24k"
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
  --wandb-run-name gsm-m-24k-f5tts \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,24k,f5tts-mel \
  --exp-dir "${EXP_ROOT}/24k"
```

## 9. 解码

解码示例：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/24k" \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

带 W&B 的版本：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/24k" \
  --max-duration 600 \
  --decoding-method modified_beam_search \
  --use-wandb True \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name gsm-m-24k-f5tts \
  --wandb-tags gigaspeech,zipformer,subset-m,no-musan,24k,f5tts-mel
```

## 10. 预期输出

重要输出包括：

- `${EXP_ROOT}/24k/tensorboard/`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-dev-*.txt`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-test-*.txt`
- `${EXP_ROOT}/24k/wandb_run_id.txt`，当启用 W&B 时会生成

## 11. 备注

- `stage 5-6` 是可选的 split-based 特征计算辅助流程，不属于主 `M` 训练路径。
- 主 Zipformer 路径默认把 `--enable-musan` 设为 `False`。
- 在 CPU-only 机器上做预处理时，请使用 `--cpu-only true`。
- 如果你正在和 `gigaspeech_16k` 做并行对比，建议使用同一个 W&B project/group，但把 `exp-dir` 分到不同子目录。
- 如果要先做 smoke test，优先缩短 epoch 数、减小 `batch-duration` 或减少 `max-duration`。

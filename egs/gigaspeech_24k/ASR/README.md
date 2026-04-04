# GigaSpeech 24k ASR

这个目录是从 `egs/gigaspeech/ASR` 派生出来的 GigaSpeech ASR recipe。
它的职责同样刻意收得比较窄，主要包括：

- 基于自定义提取器的 24 kHz F5-TTS log-mel 特征流水线
- 在特征计算阶段把原始 16 kHz 音频离线重采样到 24 kHz
- 用于对比和跟踪的 24 kHz 实验元信息
- 训练和解码阶段的可选 W&B 记录
- 对 CPU-only 数据准备更友好的入口
- 主 Zipformer 路径默认关闭 MUSAN

和 `egs/gigaspeech_16k/ASR` 相比，这个目录最核心的差异不在模型框架，而在前端特征：

- 16k recipe: 标准 Kaldifeat fbank，80 维
- 24k recipe: 自定义 F5-TTS 风格 log-mel，100 维

## 数据集

该 recipe 面向 GigaSpeech 英文 ASR 数据集：
<https://github.com/SpeechColab/GigaSpeech>

如果 `download/GigaSpeech` 还没有准备好，请先申请访问权限，再建立预期的符号链接：

```bash
ln -sfv /path/to/GigaSpeech download/GigaSpeech
```

## 依赖

该 recipe 依赖常规的 Icefall ASR 环境：

- `torch`
- `torchaudio`
- `lhotse`
- `k2`
- `sentencepiece`
- `tensorboard`
- 重新生成 BPE 资源时需要 `jq`
- 只有在使用 `--use-wandb True` 时才需要 `wandb`

其中 `torchaudio` 是这个 recipe 的关键依赖，因为 `local/f5tts_mel_extractor.py`
会在特征提取时完成 16 kHz -> 24 kHz 的重采样，并生成 100 维 log-mel 特征。

## 建议继续阅读

- [RUNBOOK.md](RUNBOOK.md)：一步一步的执行说明
- [RESULTS.md](RESULTS.md)：本 recipe 的结果记录规范和模板

## 稳定运行建议

`24k` recipe 的 stage 4 和 `16k` recipe 不同点不只是输出维度从 `80` 维变成 `100` 维。
这里的 `local/f5tts_mel_extractor.py` 还会在提特征时在线做 `16 kHz -> 24 kHz` 重采样，因此同样的 `batch-duration` 下，`24k` 的瞬时内存和累计主存压力都更大。

实践上建议：

- 不要把 `gigaspeech_24k` 的 stage 4 和 `gigaspeech_16k` 的 stage 4 放在同一个受限 cgroup 里并跑。
- 在大内存 CPU-only 机器上，优先单独完成 `24k` 的 `M` 子集特征提取。
- 在启动前同时检查宿主机内存、`/dev/shm` 和 cgroup 上限，而不是只看 `free -h`。

推荐检查命令：

```bash
free -h
df -h /dev/shm
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

如果目标实例大约有 `500 GiB` RAM 和 `500 GiB` `/dev/shm`，推荐默认参数是：

```bash
CUDA_VISIBLE_DEVICES='' bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 4 \
  --stage4-batch-duration 200
```

更稳的首跑参数是：

```bash
CUDA_VISIBLE_DEVICES='' bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 2 \
  --stage4-batch-duration 150
```

逐步提速时，优先按下面的顺序调：

- 先用 `2/150`
- 再切到 `4/200`
- 如果长时间稳定，再试 `4/250`

不建议直接从 `500` 起跑 `24k stage 4`，即使 `16k` 曾经用 `500` 成功跑完。`24k` 的在线重采样路径更重，而 `num-workers` 带来的收益通常小于大 batch 带来的风险。

## 说明

- 这个目录里的主路径仍然是基于子集 `M` 的 `zipformer` recipe。
- 当前 24k recipe 的重点是统一 100 维特征契约，确保数据准备、训练、解码和导出入口能够消费同一套 24k 特征。
- 如果要和 16k recipe 做并排对比，建议复用同一个 W&B project/group，并把两个实验目录放在同一个父目录下。
- 目前默认的数据产物仍然写到本目录下的 `data/`，后续如果要迁移到共享存储或 `public` 目录，建议统一通过脚本参数或软链处理。

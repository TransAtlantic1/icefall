# GigaSpeech 24k ASR

这个目录是从 `egs/gigaspeech/ASR` 派生出来的 GigaSpeech ASR recipe。
它的职责同样刻意收得比较窄，主要包括：

- 基于自定义提取器的 24 kHz F5-TTS log-mel 特征流水线
- 在 `prepare.sh` stage 4 把原始 recordings 离线升采样到 24 kHz
- 在 stage 5-6 基于这套 24 kHz recordings 生成 raw cuts 和 100 维特征
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
- 重新生成 BPE 资源时优先使用 `jq`；缺失时会自动回退到 Python
- 只有在使用 `--use-wandb True` 时才需要 `wandb`

其中 `torchaudio` 是这个 recipe 的关键依赖：

- `local/resample_recordings_to_flac.py` 用它把 recordings 离线升采样到 24 kHz FLAC
- `local/f5tts_mel_extractor.py` 用它计算 100 维 F5-TTS 风格 log-mel 特征

采样率方面有一个容易踩坑的点：

- GigaSpeech manifest 里的 `sampling_rate` 往往是语料语义上的 `16 kHz`
- 但原始 `.opus` 文件实际解码出来通常是 `48 kHz`
- 对 `24k` recipe 而言，推荐且现在默认遵循的原则是：直接从原始音频一次重采样到 `24 kHz`
- 不推荐先按 manifest 读成 `16 kHz`，再做 `16 -> 24 kHz` 二次重采样

因此：

- `prepare.sh` stage 4 的离线缓存是按原始 source 直接生成 `24 kHz` FLAC
- `zipformer/streaming_decode.py` 也应直接从原始 source 切段并一次重采样到 `24 kHz`
- 批量 `decode.py` 的默认路径仍然是消费预计算好的 `24 kHz` 特征

## 数据准备流程

主流程的 stage 编号如下：

- `1`: 准备 `M` / `DEV` / `TEST` manifests
- `3`: 把 `M` 的 recordings 切成分片
- `4`: 离线升采样 DEV / TEST 和 `M` 的 recording shards
- `5`: 文本预处理并基于 resampled recordings 生成 raw cuts
- `6`: 计算 DEV / TEST / M 的主特征
- `7-8`: 可选的 split-based `M` 特征计算
- `10`: 准备 `lang_bpe_500`
- `11`: 准备 phone lexicon

新增的离线升采样产物都落在本目录下的 `data/`：

- `data/manifests_resampled/24000/`
- `data/audio_cache/gigaspeech/24000/`
- `data/fbank/`

如果你要并行跑 stage 4，可以使用：

```bash
bash run_resample_shard.sh --instance-index 0
```

更完整的流程见 [RUNBOOK.md](RUNBOOK.md)。

## 独立训练入口

这个目录现在提供了可单独起跑的离线训练脚本：

```bash
bash run_train_offline.sh
```

默认行为：

- 默认使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`
- 默认把实验输出写到 `../experiments/gigaspeech_24k_train`
- 默认启用 `WANDB_MODE=offline`
- 启动前会检查 `wandb` 可用性，以及 `DEV` 特征是否真的是 `100` 维
- 如果 `24k` 特征或 `lang_bpe_500` 还没准备好，脚本会直接失败，不会误起训练

如果你要和别的训练并跑，再显式覆盖 GPU 和端口，例如：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=12364 bash run_train_offline.sh
```

## 稳定运行建议

新的 24k 流程把重负载拆成了两段：

- stage 4 离线升采样：更偏 CPU / IO / 磁盘写入
- stage 6 / 8 特征提取：更偏内存和 batch 峰值

实践上建议：

- 不要把 `gigaspeech_24k` 的 stage 4/6 和 `gigaspeech_16k` 的特征计算放在同一个受限 cgroup 里并跑。
- 大内存 CPU-only 机器上，优先单独完成 stage 4，再跑 stage 5-6。
- stage 6 优先调 `--feature-batch-duration`，其次再调 `--feature-num-workers`。
- stage 4 的并行度用 `--resample-num-workers` 控制；多机并行优先用 `run_resample_shard.sh`。

推荐检查命令：

```bash
free -h
df -h /dev/shm
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

## 说明

- 这个目录里的主路径仍然是基于子集 `M` 的 `zipformer` recipe。
- 当前 24k recipe 的重点是统一 100 维特征契约，确保数据准备、训练、解码和导出入口能够消费同一套 24k 特征。
- 对 GigaSpeech 这类 OPUS 数据，统一前端时要特别避免 `raw -> 16k -> 24k` 这种级联重采样，优先保持 `raw -> 24k` 单次重采样。
- `prepare.sh` 仍然保留了 `--stage4-num-workers` 和 `--stage4-batch-duration` 作为兼容参数，但新流程推荐使用 `--feature-num-workers` 和 `--feature-batch-duration`。
- 如果要和 16k recipe 做并排对比，建议复用同一个 W&B project/group，并把两个实验目录放在同一个父目录下。
- 目前默认的数据产物仍然写到本目录下的 `data/`，后续如果要迁移到共享存储或 `public` 目录，建议统一通过脚本参数或软链处理。
- 如果需要把离线 W&B 记录回传到联网实例，可以执行 `bash sync_wandb_offline.sh`。

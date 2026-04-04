# GigaSpeech 16k ASR

这个目录是从 `egs/gigaspeech/ASR` 派生出来的 GigaSpeech ASR recipe。
它的职责刻意收得比较窄，主要包括：

- 标准 Kaldifeat fbank 特征提取
- 用于对比和跟踪的 16 kHz 实验元信息
- 训练和解码阶段的可选 W&B 记录
- 对 CPU-only 数据准备更友好的入口
- 主 Zipformer 路径默认关闭 MUSAN

这个目录本身不引入单独的离线重采样流水线。

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

## 建议继续阅读

- [RUNBOOK.md](RUNBOOK.md)：一步一步的执行说明
- [RESULTS.md](RESULTS.md)：本 recipe 的结果记录规范和模板

## 说明

- 这个目录里的主路径是基于子集 `M` 的 `zipformer` recipe。
- 该 recipe 可以复用另一个 GigaSpeech 工作区中的 DEV/TEST 特征和 `lang_bpe_500`，也可以在本地自行生成所需资源。
- 如果要和 24k recipe 做并排对比，建议复用同一个 W&B project/group，并把两个实验目录放在同一个父目录下。

# 结果记录

这个文件只用于记录 `egs/gigaspeech_24k/ASR` 目录下实际跑出来的结果。

不要把 `egs/gigaspeech/ASR` 或 `egs/gigaspeech_16k/ASR` 的历史结果直接当作本目录结果拷贝进来。

## 当前状态

目前这个目录已经定义好了 24 kHz F5-TTS log-mel recipe、运行流程和 100 维特征契约，但这里还没有补齐一套经过统一确认的本目录实验结果。

在真正把实验结果写进来之前，请把其他目录里的数字只当成参考材料，不要当成这个目录已经验证过的输出。

## 这里应该记录什么

建议每次实验至少记录以下内容：

- 模型类型，例如 `zipformer`
- 训练子集，例如 `M`
- 特征类型，例如 `24k F5-TTS log-mel`
- 特征维度，例如 `100`
- 是否启用 MUSAN
- 训练命令
- 解码命令
- checkpoint 选择方式，例如 `--epoch 30 --avg 15`
- 最终 `dev` 和 `test` WER
- 可选的 W&B 链接或实验目录

## 建议模板

````markdown
### zipformer

- 日期：
- 训练子集：
- 特征类型：24k F5-TTS log-mel
- 特征维度：100
- MUSAN：
- 实验目录：
- W&B：

| 解码方式 | dev | test | 备注 |
|----------|-----|------|------|
| modified_beam_search |     |      |      |

训练命令：
```bash
python zipformer/train.py \
  ...
```

解码命令：
```bash
python zipformer/decode.py \
  ...
```
````

## 建议记录方式

- 如果一次实验只调整了前端或训练参数中的一项，也请把完整命令和实验目录一起写下来。
- 如果训练和解码都启用了 W&B，建议把同一个 run 或 group 一并记录下来，方便和 `gigaspeech_16k` 做并排比较。
- 如果当前结果只验证了 `zipformer`，请不要把 `conformer_ctc` 或 `pruned_transducer_stateless2` 的旧数字直接补到这里。

## 参考说明

这个 recipe 本质上是 GigaSpeech 24 kHz 特征对比实验目录，重点在于：

- 16 kHz 音频离线重采样到 24 kHz
- 使用 100 维 F5-TTS 风格 log-mel 特征
- 统一训练、解码、导出入口对这套特征的消费方式

如果你需要查看上游 baseline 或 16k 对照实验，可以参考：

- `egs/gigaspeech/ASR/RESULTS.md`
- `egs/gigaspeech_16k/ASR/RESULTS.md`

除非这些结果在本目录下重新跑过，或者已经明确验证过适用于本目录，否则不要直接复制到这里。

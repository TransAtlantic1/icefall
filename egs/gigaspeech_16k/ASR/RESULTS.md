# 结果记录

这个文件只用于记录 `egs/gigaspeech_16k/ASR` 目录下实际跑出来的结果。

不要把 `egs/gigaspeech/ASR` 的历史结果直接当作本目录结果拷贝进来。

## 当前状态

目前这个目录已经定义好了 recipe 和运行流程，但这里还没有补齐一套经过确认的本目录实验结果。

在真正把实验结果写进来之前，请把上游 GigaSpeech 的数字只当成参考材料，不要当成这个目录已经验证过的输出。

## 这里应该记录什么

建议每次实验至少记录以下内容：

- 模型类型，例如 `zipformer`
- 训练子集，例如 `M`
- 特征类型，例如 `Kaldifeat fbank`
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
- 特征类型：
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

## 参考说明

这个 recipe 本质上是标准 GigaSpeech Kaldifeat fbank 流水线的一个 16 kHz 元信息/W&B 变体。
如果你需要查看历史上游 baseline，可以参考：

- `egs/gigaspeech/ASR/RESULTS.md`

除非这些结果在本目录下重新跑过，或者已经明确验证过适用于本目录，否则不要直接复制到这里。

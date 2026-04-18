# 结果记录

这个文件只记录 `egs/gigaspeech_16k/ASR` 在本目录 recipe 下重新跑出的结果。

建议每次记录：

- 日期
- 模型类型
- 训练子集
- 特征类型与维度
- 训练命令
- 解码命令
- checkpoint 选择方式
- `dev/test` 指标
- 实验目录或 W&B 链接

模板：

````markdown
### zipformer

- 日期：
- 训练子集：`M`
- 特征：`80-dim fbank`
- 实验目录：

| 解码方式 | dev | test | 备注 |
|----------|-----|------|------|
| greedy_search |     |      |      |

训练命令：
```bash
python zipformer/train.py ...
```

解码命令：
```bash
python zipformer/decode.py ...
```
````

历史 smoke / 迁移记录不再堆叠在这里；如果需要追溯旧文本副本，查看 `asr_op/giagspeech/backup/`。

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

## 本机 smoke 结果

### zipformer 2x48G local smoke

- 日期：2026-04-06
- 训练子集：`M`
- 特征类型：`Kaldifeat fbank`
- 特征维度：`80`
- MUSAN：`False`
- 机器：`2 x 48 GiB GPU`
- 目标迁移机器：`8 x H200`，单卡约 `141 GiB`
- 统一设置：`FP16`，`smoke_num_batches=8`，`small-dev=true`

实测结论：

- `--print-diagnostics true` 不能替代 smoke。`FP16 + diagnostics` 会在诊断收尾阶段因为 `torch.linalg_eigh/eig` 的 half precision 限制失败
- `--num-workers 1` 在本机会触发 DataLoader bus error，根因是 `/dev/shm` 不足
- 因此本地 smoke 包装脚本默认改成 `--num-workers 0`

已通过的本机 smoke：

| max-duration | num-workers | peak reserved / GPU | avg step time | 实验目录 |
|--------------|-------------|---------------------|---------------|----------|
| 700 | 0 | 17.896 GiB | 1.048 s | `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_smoke/20260405-2x48g/16k_md700_nw0` |
| 1600 | 0 | 35.188 GiB | 1.778 s | `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_smoke/20260405-2x48g/16k_md1600_nw0` |
| 2000 | 0 | 43.498 GiB | 1.903 s | `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_smoke/20260405-2x48g/16k_md2000_nw0` |

推荐的 H200 直接起训配置：

- `WORLD_SIZE=8`
- `USE_FP16=1`
- `MAX_DURATION=2000`
- `--num-workers 0`

本次用于近上限校准的命令：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall

./run_smoke_train_offline.sh \
  --gpus 0,1 \
  --exp-root /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_smoke/20260405-2x48g/16k_md2000_nw0 \
  --data-root /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data \
  --master-port 12385 \
  --max-duration 2000 \
  --smoke-num-batches 8 \
  --num-workers 0 \
  --use-fp16 1
```

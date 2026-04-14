# Emilia 24k 中文 Batch Size 与学习率调度分析报告

本报告只整理当前仓库中的代码事实和本地实验产物，不修改任何训练代码。

分析目标有两个：

- 说明 `gigaspeech_16k` 与 `gigaspeech_24k` 的 `batch size / max_duration / learning rate` 之间到底是什么关系。
- 把这个结论落回 `icefall/egs/emilia_24k_multilang/emilia_24k_ZH`，给后续调参一个可直接执行的依据。

## 1. 执行摘要

结论先说。

- `gigaspeech_16k` 和 `gigaspeech_24k` 两份 `train.py` 在学习率调度逻辑上基本一致，核心都是 `ScaledAdam + Eden`，默认 `base_lr=0.045`、`lr_batches=7500`、`lr_epochs=1`。
- 两份 recipe 的主要差异不是 scheduler 参数，而是输入特征：`16k` 用 `kaldifeat_fbank`、`feature_dim=80`；`24k` 用 `f5tts_mel`、`feature_dim=100`。
- 真正造成 batch size 与 lr 曲线明显变化的不是 `16k/24k` 切换，而是 `max_duration` 从 `4000` 改到 `1000`。
- `max_duration` 改小后，sampler 每步打包的总时长变小，典型 batch size 约缩小到原来的 `1/4`，对应的 `steps per epoch` 约增大到 `4` 倍。
- 但是 `Eden` 学习率调度仍按原始 `batch_idx_train` 计步，不按“累计音频时长”计步，所以 `max_duration=1000` 时会更快完成 warmup，也会在相同 epoch 下更快进入衰减。
- 与之相对，模型内部很多 `ScheduledFloat` 是通过 `get_adjusted_batch_count()` 按 `max_duration * world_size / ref_duration` 做了时长归一化。这意味着模型内部 schedule 与优化器 lr schedule 不在同一个“数据进度”坐标系上。
- 因此，`max_duration=1000` 的 lr 曲线不是简单地“整体更小”或“整体更大”，而是更快冲到峰值，然后更快掉下去。
- 如果目标是在更小 `max_duration` 下尽量复现 `max_duration=4000` 的 lr-vs-data 行为，优先应该放大 step-based scheduler 参数，而不是先降低 `base_lr`。

## 2. 代码依据

本报告的代码依据主要来自：

- `icefall/egs/gigaspeech_16k/ASR/zipformer/train.py`
- `icefall/egs/gigaspeech_24k/ASR/zipformer/train.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/train.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/optim.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/asr_datamodule.py`

本地实验依据主要来自：

- `experiments/gigaspeech_h200/16k_train_g0-3/...`
- `experiments/gigaspeech_h200/24k_train_g4-7/...`
- 其中的 `wandb config.yaml`
- `wandb-summary.json`
- `output.log`
- TensorBoard `events.out.tfevents.*`
- `launch.log`

## 3. 两份 GigaSpeech `train.py` 的学习率相关逻辑

### 3.1 调度骨架一致

`gigaspeech_16k` 与 `gigaspeech_24k` 的训练脚本在学习率相关逻辑上是一致的：

- 默认 `--base-lr 0.045`
- 默认 `--lr-batches 7500`
- 默认 `--lr-epochs 1`
- 优化器是 `ScaledAdam`
- scheduler 是 `Eden(optimizer, params.lr_batches, params.lr_epochs)`
- 训练日志和 TensorBoard / W&B 记录的都是 `train/learning_rate`

这说明 `16k` 和 `24k` 并没有各自定义一套不同的 lr schedule。

### 3.2 两份脚本的主要差异不在 scheduler

直接比对两份 `train.py`，核心差异只有：

| 项目 | `gigaspeech_16k` | `gigaspeech_24k` |
| --- | --- | --- |
| `recipe` | `gigaspeech_16k` | `gigaspeech_24k` |
| `feature_type` | `kaldifeat_fbank` | `f5tts_mel` |
| `feature_sample_rate` | `16000` | `24000` |
| `feature_dim` | `80` | `100` |

学习率调度参数、训练主循环、scheduler 调用方式没有实质差异。

## 4. `max_duration` 同时影响了 batch size 和 lr 进度

### 4.1 sampler 直接使用 `max_duration`

在 datamodule 中，训练 sampler 直接用：

- `DynamicBucketingSampler(..., max_duration=self.args.max_duration, ...)`

这意味着 `max_duration` 本质上控制的是“单步允许打包的总时长预算”。

当 `max_duration` 从 `4000` 下降到 `1000` 时，单步总时长预算缩小到原来的 `1/4`，因此单步能容纳的 utterance 数量也会明显下降，steps per epoch 则相应上升。

### 4.2 模型内部 schedule 做了时长归一化

训练脚本里有：

```python
def get_adjusted_batch_count(params):
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )
```

并且训练过程中会周期性调用：

```python
set_batch_count(model, get_adjusted_batch_count(params))
```

所以模型内部依赖 `batch_count` 的很多 `ScheduledFloat`，实际看到的是一个按时长归一化后的进度，而不是裸的 step 数。

### 4.3 优化器 lr schedule 没做同样的归一化

`Eden` 的公式是：

```text
lr = base_lr
     * ((batch^2 + lr_batches^2) / lr_batches^2)^(-0.25)
     * ((epoch^2 + lr_epochs^2) / lr_epochs^2)^(-0.25)
     * warmup
```

其中：

- `batch` 是原始训练 step 计数
- `epoch` 是原始训练轮数
- warmup 默认按 `500` 个 batch 线性从 `0.5` 升到 `1.0`

这里没有用 `get_adjusted_batch_count()`，所以：

- sampler 看到的是按时长预算变化后的 step 数
- 模型内部 schedule 看到的是归一化后的“时长进度”
- lr scheduler 看到的却仍是原始 step 数

这就是当前 batch size / lr 问题的根因。

## 5. 本地两组实验的实际对应关系

本地 `experiments/gigaspeech_h200` 中，`16k` 和 `24k` 都存在两组主训练：

| 任务 | 配置 | 本地 run 时间 | run id |
| --- | --- | --- | --- |
| 16k | `max_duration=4000`, `num_epochs=30` | `2026-04-06 12:44:47 UTC` | `glt62m94` |
| 16k | `max_duration=1000`, `num_epochs=40` | `2026-04-08 02:07:36 UTC` | `glt62m94` |
| 24k | `max_duration=4000`, `num_epochs=30` | `2026-04-06 12:45:00 UTC` | `sy372le8` |
| 24k | `max_duration=1000`, `num_epochs=40` | `2026-04-08 02:22:22 UTC` | `sy372le8` |

对应的 `wandb config.yaml` 明确记录了这四组配置。

需要特别注意：

- 第二次 `max_duration=1000` 的训练不是从 `4000` 那次 checkpoint 接着训。
- 本地 `config.yaml` 中写的是 `--start-epoch 1`，所以它是新的从头训练。
- 但是同一个 `exp-dir` 下会复用 `wandb_run_id.txt`，因此两个不同配置使用了同一个 run id。

## 6. W&B 记录为什么容易误读

训练脚本会把 run id 写到：

- `exp-dir/wandb_run_id.txt`

下次同目录启动时，如果 `--wandb-run-id` 没显式传入，就会读这个文件并复用 id。

而本地 `launch.log` 又明确出现了：

- `resume will be ignored since W&B syncing is set to offline`
- `Starting a new run with run id glt62m94`
- `Starting a new run with run id sy372le8`

这意味着：

- 离线模式下，W&B 并没有真正意义上的在线 resume
- 但本地和后续 sync 仍沿用了同一个 run id

因此，W&B 网页上一条 run 很可能混入了不同配置、不同启动批次的历史。分析参数行为时，应该优先以本地拆开的 TensorBoard event 文件和各自的 `config.yaml` 为准，而不能把网页上的单条 run 直接当成“单一配置的完整连续历史”。

## 7. 实际 batch size、steps per epoch 与 lr 变化

### 7.1 `steps per epoch`

从本地 `wandb-summary.json` 提取：

| 配置 | final batch idx | epoch 数 | steps/epoch |
| --- | ---: | ---: | ---: |
| `16k md4000` | `7110` | `30` | `237.0` |
| `16k md1000` | `38200` | `40` | `955.0` |
| `24k md4000` | `7410` | `30` | `247.0` |
| `24k md1000` | `40029` | `40` | `1000.7` |

结论很直接：`max_duration=1000` 时，每个 epoch 的 step 数约是 `max_duration=4000` 的 `4` 倍。

### 7.2 `train/batch_size` 统计

从 TensorBoard 标量提取：

| 配置 | 记录点数 | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `16k md4000` | `142` | `858.77` | `676` | `200` | `2051` |
| `16k md1000` | `764` | `220.00` | `189` | `50` | `512` |
| `24k md4000` | `148` | `856.69` | `757` | `200` | `2051` |
| `24k md1000` | `800` | `220.86` | `189` | `50` | `512` |

这里也能看到非常稳定的规律：

- `16k` 和 `24k` 的 batch size 统计几乎重合
- `max_duration` 才是主要变量
- `4000 -> 1000` 后，典型 batch size 大约缩小到 `1/4`

### 7.3 `train/learning_rate` 的实际趋势

从拆开的 TensorBoard event 文件恢复 `train/learning_rate`，代表点如下。

#### `16k md4000`

| 训练进度 | lr |
| --- | ---: |
| step `50` | `0.02474972` |
| step `500` | `0.03005997` |
| step `1000` | `0.02206414` |
| step `2000` | `0.01557850` |
| step `4000` | `0.01055724` |
| final step `7100` | `0.00711893` |

#### `16k md1000`

| 训练进度 | lr |
| --- | ---: |
| step `50` | `0.02474972` |
| step `500` | `0.04495014` |
| step `1000` | `0.03767400` |
| step `2000` | `0.02958090` |
| step `4000` | `0.02081716` |
| step `8000` | `0.01310671` |
| step `16000` | `0.00732208` |
| step `24000` | `0.00491335` |
| step `32000` | `0.00374115` |
| final step `38200` | `0.00316229` |

#### `24k md4000`

| 训练进度 | lr |
| --- | ---: |
| step `50` | `0.02474972` |
| step `500` | `0.03005997` |
| step `1000` | `0.02206414` |
| step `2000` | `0.01557850` |
| step `4000` | `0.01055724` |
| final step `7400` | `0.00704814` |

#### `24k md1000`

| 训练进度 | lr |
| --- | ---: |
| step `50` | `0.02474972` |
| step `500` | `0.04495014` |
| step `1000` | `0.04480219` |
| step `2000` | `0.03719601` |
| step `4000` | `0.02377025` |
| step `8000` | `0.01399521` |
| step `16000` | `0.00756119` |
| step `24000` | `0.00512215` |
| step `32000` | `0.00385983` |
| step `40000` | `0.00309284` |

### 7.4 按 epoch 对齐看 lr 更清楚

只看 step 会掩盖一个事实：`max_duration=1000` 时，同一 epoch 已经包含更多 step。

按 epoch 起点附近的第一个 lr 点对齐：

| 配置 | epoch 2 | epoch 5 | epoch 10 | epoch 20 | epoch 30 | epoch 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `16k md4000` | `0.02837238` | `0.02207355` | `0.01466170` | `0.00953916` | `0.00716643` | - |
| `16k md1000` | `0.03767400` | `0.02090285` | `0.01212352` | `0.00637546` | `0.00427065` | `0.00320082` |
| `24k md4000` | `0.02837238` | `0.02206414` | `0.01463533` | `0.00949664` | `0.00709527` | - |
| `24k md1000` | `0.04480219` | `0.02078827` | `0.01194538` | `0.00624415` | `0.00417673` | `0.00312893` |

这张表说明：

- 在早期，`md1000` 因为 step 走得更快，会更早完成 warmup，所以 lr 反而更高。
- 进入中后期后，`md1000` 在同一个 epoch 下已经积累了更多 step，因此 lr 会比 `md4000` 更低。
- 所以 `md1000` 的真实趋势是“更快升到峰值，再更快掉下去”。

## 8. 根因分析

把现象压缩成一句话：

> `max_duration` 改小以后，数据打包尺度按时长变小了，但 lr scheduler 仍按原始 step 数行走，导致 lr-vs-data 的节奏发生系统性偏移。

可以拆成三层：

1. `max_duration=1000` 让每步处理的数据时长变成原来的约 `1/4`
2. 同一份数据集因此需要约 `4` 倍 step 才能走完一个 epoch
3. `Eden` 的 warmup 和衰减都按 step 计数，因此：
   - warmup 在“数据时长尺度”上更短
   - 衰减在“epoch / 数据遍历尺度”上更快

这就是为什么只看 batch size 会不够，只看 lr 末值也会误判。

## 9. 对 Emilia 24k ZH 的直接建议

### 9.1 短期参数建议

如果后续 `emilia_24k_ZH` 也采用更小的 `max_duration`，例如从 `4000` 级别降到 `1000` 级别，而目标仍然是尽量保持与原来接近的 lr-vs-data 行为，那么优先建议是：

- 先不要机械降低 `base_lr`
- 优先把 step-based scheduler 参数按相同比例放大

按照 `4000 -> 1000` 这个 `4x` 关系，第一版可尝试：

| 参数 | 原值 | 建议起点 |
| --- | ---: | ---: |
| `base_lr` | `0.045` | `0.045` |
| `lr_batches` | `7500` | `30000` |
| `warmup_batches` | `500` | `2000` |
| `lr_epochs` | `1` | 先保持 `1` |

原因是：

- `md1000` 当前已经在中后期比 `md4000` 衰减得更快
- 如果这时再先降 `base_lr`，等于叠加了第二重保守
- 这种改法更接近“保持 lr 与累计数据量的对应关系”

### 9.2 长期实现建议

如果允许对训练代码做小改动，更干净的方案不是无限调 `lr_batches`，而是让 lr scheduler 也基于 duration-adjusted progress。

也就是说，可以考虑让 `Eden` 使用与模型内部 `ScheduledFloat` 一致的“归一化 batch count”，而不是原始 `batch_idx_train`。

这样做的好处是：

- `max_duration` 的变化不再同时改写模型内部 schedule 与优化器 schedule 的相对节奏
- 调整 batch packing 主要影响吞吐与显存，而不再隐式改变 lr-vs-data 曲线

## 10. 最终结论

对当前这批 `GigaSpeech` 实验，最重要的判断不是“更小 batch 要不要更小学习率”，而是：

- 更小 `max_duration` 已经让有效训练进度坐标发生了变化
- 它同时改变了 batch size、steps per epoch 和 lr scheduler 的相对节奏
- 因此如果只改 `max_duration` 而不改 scheduler，得到的不是同一条训练轨迹的简单缩放版

对 `Emilia 24k ZH` 来说，这个结论的直接含义是：

- 以后调 `max_duration` 时，必须把 lr schedule 当成联动项一起看
- 如果目标是稳定复现已有训练行为，优先联动 `lr_batches` 和 warmup，而不是先动 `base_lr`
- 如果目标是把 recipe 设计得更稳健，长期应该让 lr 调度与 duration-adjusted progress 对齐

## 附录 A. 关键文件

代码文件：

- `icefall/egs/gigaspeech_16k/ASR/zipformer/train.py`
- `icefall/egs/gigaspeech_24k/ASR/zipformer/train.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/train.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/optim.py`
- `icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/zipformer/asr_datamodule.py`

实验文件：

- `experiments/gigaspeech_h200/16k_train_g0-3/wandb_offline/wandb/offline-run-20260406_124447-glt62m94/files/config.yaml`
- `experiments/gigaspeech_h200/16k_train_g0-3/wandb_offline/wandb/offline-run-20260408_020736-glt62m94/files/config.yaml`
- `experiments/gigaspeech_h200/24k_train_g4-7/wandb_offline/wandb/offline-run-20260406_124500-sy372le8/files/config.yaml`
- `experiments/gigaspeech_h200/24k_train_g4-7/wandb_offline/wandb/offline-run-20260408_022222-sy372le8/files/config.yaml`
- `experiments/gigaspeech_h200/16k_train_g0-3/zipformer_m_g0-1-2-3/tensorboard/events.out.tfevents.1775479486.fangjie-emilia-asr-train--43267e97869a-sueemxptkq.3290794.0`
- `experiments/gigaspeech_h200/16k_train_g0-3/zipformer_m_g0-1-2-3/tensorboard/events.out.tfevents.1775614056.fangjie-emilia-asr-train--43267e97869a-63edfyq4zv.280711.0`
- `experiments/gigaspeech_h200/24k_train_g4-7/zipformer_m_g4-5-6-7/tensorboard/events.out.tfevents.1775479500.fangjie-emilia-asr-train--43267e97869a-sueemxptkq.3293354.0`
- `experiments/gigaspeech_h200/24k_train_g4-7/zipformer_m_g4-5-6-7/tensorboard/events.out.tfevents.1775614942.fangjie-emilia-asr-train--43267e97869a-63edfyq4zv.430880.0`
- `experiments/gigaspeech_h200/16k_train_g0-3/launch.log`
- `experiments/gigaspeech_h200/24k_train_g4-7/launch.log`

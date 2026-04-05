# GigaSpeech 24k ASR 运行手册

标准主流程脚本：`prepare.sh`。

本 recipe 的特征配置：
- 原始音频主要仍然是 16 kHz
- 先把 recordings 离线升采样到 24 kHz FLAC 缓存
- 再基于 24 kHz recordings 生成 raw cuts 和 100 维特征
- 默认主流程：`stage 1 -> 3 -> 4 -> 5 -> 6`
- `stage 7-8` 是可选的 split-based `M` 子集特征链路

数据准备可以在 CPU-only 机器上完成。训练和解码建议在 GPU 机器上完成。

## 1. 环境

所有命令都在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR
```

推荐把常用环境变量合并成下面这一段，直接复制即可：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_24k_train
export WANDB_PROJECT=gigaspeech-24k
export WANDB_GROUP=zipformer-m
```

如果当前机器是 CPU-only 机器，不要运行 `train.py` 或 `decode.py --help`，因为 GPU 版 `k2` 可能会在导入时尝试加载 CUDA 相关库。训练和解码相关检查建议只在真正的 GPU 训练机上做。

## 2. Stage 对照表

`prepare.sh` 的阶段划分如下：

1. Stage 0：下载 GigaSpeech 和 MUSAN
2. Stage 1：生成 `M/DEV/TEST` manifests
3. Stage 2：准备 MUSAN manifest
4. Stage 3：把 `M` recordings 切成 shards
5. Stage 4：离线把 recordings 升采样到 24 kHz
6. Stage 5：文本预处理并基于 recordings 生成 raw cuts
7. Stage 6：计算 `DEV/TEST/M` 的 100 维特征
8. Stage 7：把 `M` raw cuts 切成 split manifests
9. Stage 8：按 split 重新计算 `M` 的特征
10. Stage 9：计算 MUSAN 特征
11. Stage 10：准备 BPE 语言目录
12. Stage 11：准备 phone lexicon

重要默认参数：
- `stage=1`
- `stop_stage=6`
- `recording_num_splits=100`
- `resample_num_workers=32`
- `target_sample_rate=24000`
- `use_resampled_audio=true`
- `stage4_num_workers=32`
- `stage4_batch_duration=1000`

如果你想优先稳定完成 `24k` 主特征，当前推荐参数是：
- `--feature-num-workers 4`
- `--feature-batch-duration 200`

## 3. 按 Stage 与设备划分的运行方式

先按用途把 stage 分清楚，再按 CPU/GPU 选命令：

| Stage | 内容 | 推荐设备 | 并行方式 |
|---|---|---|---|
| 1 | 生成 manifests | CPU | 单机一次 |
| 2 | 准备 MUSAN manifests | CPU | 单机一次，可选 |
| 3 | 切分 `M` recordings | CPU | 单机一次 |
| 4 | 离线 24k 重采样 | CPU | 多实例分片 |
| 5 | 文本预处理 + raw cuts | CPU | 单机一次 |
| 6 | 计算主 24k 特征 | CPU | 单机一次 |
| 7 | 切分 `M` raw cuts | CPU | 单机一次，可选 |
| 8 | 计算 split `M` 特征 | CPU 或 GPU | 多分片，可选 |
| 9 | MUSAN 特征 | CPU | 单机一次，可选 |
| 10 | BPE | CPU | 单机一次 |
| 11 | phone lexicon | CPU | 单机一次，可选 |

默认建议：
- Stage 1、3、4、5、6：按 CPU 跑，这是 `24k` 的标准主路径
- Stage 4：适合多实例分片并行
- Stage 7、8：只有在你明确需要 split-based `M` 特征链路时才跑
- Stage 10：训练前需要准备好
- 训练和解码：切到 GPU 机器再跑

## 4. Stage 1：准备 manifests

这一段是 CPU 流程。

### 4.1 最简单的单机跑法

```bash
bash prepare.sh \
  --stage 1 \
  --stop-stage 1 \
  --cpu-only true
```

## 5. Stage 3-4：切分 recordings 与离线升采样

这一段全部是 CPU 流程。

### 5.1 Stage 3：先切分 M recordings

```bash
bash prepare.sh \
  --stage 3 \
  --stop-stage 3 \
  --cpu-only true
```

### 5.2 Stage 4：单机离线升采样

```bash
bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --resample-num-workers 24
```

默认产物路径：
- `data/manifests_resampled/24000/gigaspeech_recordings_DEV.jsonl.gz`
- `data/manifests_resampled/24000/gigaspeech_recordings_TEST.jsonl.gz`
- `data/manifests_resampled/24000/recordings_M_split_100/`
- `data/audio_cache/gigaspeech/24000/`

### 5.3 Stage 4：多实例 CPU 分片跑法

如果你要并行做离线升采样，可以直接使用 [run_resample_shard.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR/run_resample_shard.sh)。

4 实例示例：

```bash
./run_resample_shard.sh --instance-index 0
./run_resample_shard.sh --instance-index 1
./run_resample_shard.sh --instance-index 2
./run_resample_shard.sh --instance-index 3
```

在 `recording_num_splits=100`、`num_instances=4` 时，区间是：
- worker 0：`[0, 25)`
- worker 1：`[25, 50)`
- worker 2：`[50, 75)`
- worker 3：`[75, 100)`

只有 worker 0 会顺带处理 `DEV/TEST` 的重采样。

默认日志文件写到：

```text
data/logs/resample.<start>-<end>.log
```

## 6. Stage 5-8：raw cuts 与 24k 特征提取

`stage 5-6` 是 `24k` 的标准主路径，`stage 7-8` 则是可选的 split-based `M` 特征链路。

### 6.1 Stage 5-6：标准主路径

推荐直接一次跑完：

```bash
CUDA_VISIBLE_DEVICES='' \
bash prepare.sh \
  --stage 5 \
  --stop-stage 6 \
  --cpu-only true \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

如果你想拆开执行：

```bash
bash prepare.sh \
  --stage 5 \
  --stop-stage 5 \
  --cpu-only true
```

```bash
CUDA_VISIBLE_DEVICES='' \
bash prepare.sh \
  --stage 6 \
  --stop-stage 6 \
  --cpu-only true \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

参数设计原则：
- `feature-batch-duration` 是峰值内存的主开关，优先调它
- `feature-num-workers` 影响音频读取和 DataLoader 预取
- 如果首要目标是稳定，优先把 `feature-num-workers` 控制在 `2-4`
- 在 CPU-only 机器上建议显式清空 `CUDA_VISIBLE_DEVICES`

监控建议：

```bash
watch -n 10 free -h
watch -n 10 'cat /sys/fs/cgroup/memory.current; echo; cat /sys/fs/cgroup/memory.events'
```

### 6.2 Stage 7：先切分 M raw cuts

```bash
bash prepare.sh \
  --stage 7 \
  --stop-stage 7 \
  --cpu-only true \
  --feature-num-splits 100
```

### 6.3 Stage 8：按分片重算 M 特征

只跑一部分分片的示例：

```bash
CUDA_VISIBLE_DEVICES='' \
bash prepare.sh \
  --stage 8 \
  --stop-stage 8 \
  --cpu-only true \
  --feature-num-splits 100 \
  --feature-start 0 \
  --feature-stop 25 \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

这里的 `feature-start/feature-stop` 是左闭右开区间，用来控制当前 worker 实际处理哪些 split。

## 7. Stage 9-11：MUSAN、BPE、phone

这一段全部是 CPU 流程。

### 7.1 可选 MUSAN

如果你需要 MUSAN，先准备 manifest：

```bash
bash prepare.sh \
  --stage 2 \
  --stop-stage 2
```

然后计算 MUSAN 特征：

```bash
bash prepare.sh \
  --stage 9 \
  --stop-stage 9
```

### 7.2 标准 BPE

训练前通常需要跑：

```bash
bash prepare.sh \
  --stage 10 \
  --stop-stage 10 \
  --cpu-only true
```

### 7.3 可选 phone lexicon

如果你还需要 phone based lang：

```bash
bash prepare.sh \
  --stage 11 \
  --stop-stage 11
```

## 8. 预期产物

核心输出：
- `data/manifests/gigaspeech_{recordings,supervisions}_{M,DEV,TEST}.jsonl.gz`
- `data/manifests_resampled/24000/`
- `data/audio_cache/gigaspeech/24000/`
- `data/fbank/gigaspeech_cuts_{M,DEV,TEST}.jsonl.gz`
- `data/fbank/gigaspeech_feats_{M,DEV,TEST}.lca`
- `data/fbank/gigaspeech_M_split/`
- `data/lang_bpe_500/`

## 9. 训练

当 `24k` 数据准备完成后，优先直接用 [run_train_offline.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR/run_train_offline.sh)：

```bash
bash run_train_offline.sh
```

它默认会：
- 使用 `4` 张卡：`0,1,2,3`
- 把实验写到 `../experiments/gigaspeech_24k_train`
- 默认 `WANDB_MODE=offline`
- 在真正启动训练前检查 `wandb` 和 `100` 维特征契约

如果之后你要和其他训练并跑，再显式覆盖 GPU 和端口：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=12364 bash run_train_offline.sh
```

手动多卡训练示例：

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

## 10. 解码

解码示例：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/24k" \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

## 11. 快速检查

检查预计算特征维度：

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

## 12. 备注

- `stage 5-6` 是 `24k` 的标准主路径，训练前至少要把这一段跑完
- `stage 7-8` 只是可选的 split-based `M` 特征链路，不是主训练路径的硬前置
- `use_resampled_audio=true` 时，`stage 5` 会读取 `stage 4` 生成的 24 kHz recordings manifests
- datamodule 默认把 `--enable-musan` 设为 `False`


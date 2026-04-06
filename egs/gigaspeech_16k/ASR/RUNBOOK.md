# GigaSpeech 16k ASR 运行手册

标准主流程脚本：`prepare.sh`。

本 recipe 的特征配置：
- 原始音频采样率：16 kHz
- 声学特征：Kaldifeat fbank
- 训练输入特征维度：80
- 默认主流程：`stage 1 -> 3 -> 4`
- `stage 5-6` 是可选的 split-based `M` 子集特征链路

数据准备可以在 CPU-only 机器上完成。训练和解码建议在 GPU 机器上完成。

## 1. 环境

所有命令都在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR
```

推荐把常用环境变量合并成下面这一段，直接复制即可：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_16k_train
export WANDB_PROJECT=gigaspeech-16k
export WANDB_GROUP=zipformer-m
```

如果你准备直接用 [run_train_offline.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/run_train_offline.sh)，它默认也会把实验目录写到：

```bash
EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_16k_train
```

这个 recipe 通常假设 `download/` 已经提前准备好，或者已经符号链接到本地 GigaSpeech / LM / MUSAN 数据目录，所以日常使用一般从 `stage 1` 开始，而不是从 `stage 0` 下载开始。

## 2. Stage 对照表

`prepare.sh` 的阶段划分如下：

1. Stage 0：下载 GigaSpeech 和 MUSAN
2. Stage 1：生成 `M/DEV/TEST` manifests
3. Stage 2：准备 MUSAN manifest
4. Stage 3：文本预处理并生成 raw cuts
5. Stage 4：计算 `DEV/TEST/M` 的 80 维 fbank 特征
6. Stage 5：把 `M` subset 的 raw cuts 切成 split manifests
7. Stage 6：按 split 重新计算 `M` 的特征
8. Stage 7：计算 MUSAN 特征
9. Stage 8：准备 BPE 语言目录
10. Stage 9：准备 phone lexicon

重要默认参数：
- `stage=1`
- `stop_stage=4`
- `stage4_num_workers=32`
- `stage4_batch_duration=1000`
- `cpu_only=false`
- `start=0`
- `stop=-1`

针对 `stage 4`，当前已经验证稳定完成 `M` 子集的一组参数是：
- `--stage4-num-workers 0`
- `--stage4-batch-duration 500`

这组参数的目标是优先稳定完成 `M`，而不是把音频读取并发拉高。

## 3. 按 Stage 与设备划分的运行方式

先按用途把 stage 分清楚，再按 CPU/GPU 选命令：

| Stage | 内容 | 推荐设备 | 并行方式 |
|---|---|---|---|
| 1 | 生成 manifests | CPU | 单机一次 |
| 2 | 准备 MUSAN manifests | CPU | 单机一次，可选 |
| 3 | 文本预处理 + raw cuts | CPU | 单机一次 |
| 4 | 计算主 fbank 特征 | CPU | 单机一次 |
| 5 | 切分 `M` raw cuts | CPU | 单机一次，可选 |
| 6 | 计算 split `M` 特征 | CPU 或 GPU | 多分片，可选 |
| 7 | MUSAN 特征 | CPU | 单机一次，可选 |
| 8 | BPE | CPU | 单机一次 |
| 9 | phone lexicon | CPU | 单机一次，可选 |

默认建议：
- Stage 1、3、4：按 CPU 跑，这是主 `M` 训练路径
- Stage 5、6：只有在你确实要 split-based `M` 特征链路时才跑
- Stage 8：训练前需要准备好，除非你直接复用共享的 `lang_bpe_500`
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

### 4.2 可选：复用共享的 DEV/TEST 特征和 BPE

如果你已经有一个准备好的 `egs/gigaspeech/ASR` 工作区，可以直接复用其中的 `DEV/TEST` 特征和 `lang_bpe_500`：

```bash
export SOURCE_GIGASPEECH_ASR=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech/ASR

mkdir -p data/fbank

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_DEV.jsonl.gz" data/fbank/
ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_DEV.lca" data/fbank/

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_cuts_TEST.jsonl.gz" data/fbank/
ln -sf "${SOURCE_GIGASPEECH_ASR}/data/fbank/gigaspeech_feats_TEST.lca" data/fbank/

ln -sf "${SOURCE_GIGASPEECH_ASR}/data/lang_bpe_500" data/lang_bpe_500
```

如果你不复用这些共享产物，就继续按下面的 stage 正常生成。

## 5. Stage 3-4：预处理与主特征提取

这一段全部是 CPU 流程，也是 `16k` 的标准主路径。

### 5.1 推荐的稳定跑法

```bash
bash prepare.sh \
  --stage 3 \
  --stop-stage 4 \
  --cpu-only true \
  --stage4-num-workers 0 \
  --stage4-batch-duration 500
```

等价的拆分跑法：

```bash
python local/preprocess_gigaspeech.py --cpu-only true
touch data/fbank/.preprocess_complete
python local/compute_fbank_gigaspeech.py \
  --num-workers 0 \
  --batch-duration 500
```

如果你是在共享实例上做这一步，建议同时观察宿主机内存、`/dev/shm` 和 cgroup 限额：

```bash
free -h
df -h /dev/shm
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

参数设计原则：
- `batch-duration` 是峰值内存的主开关，优先调它
- `num-workers 0` 可以减少 DataLoader 预取和共享内存波动
- 如果未来要提速，优先尝试把 `num-workers` 提到 `2`
- 不建议让 `16k stage 4` 和 `24k stage 4/6` 在同一个受限 cgroup 中并跑

## 6. Stage 5-6：可选的 split-based M 特征链路

`stage 5-6` 不是标准 `M` 训练路径。默认 Zipformer `M` 训练直接消费 `data/fbank/gigaspeech_cuts_M.jsonl.gz`，所以只有在你明确需要按 split 方式重算 `M` 特征时才跑这一段。

### 6.1 Stage 5：先切分 M raw cuts

```bash
bash prepare.sh \
  --stage 5 \
  --stop-stage 5
```

### 6.2 Stage 6：按分片重算 M 特征

跑全部分片：

```bash
bash prepare.sh \
  --stage 6 \
  --stop-stage 6
```

只跑部分分片：

```bash
bash prepare.sh \
  --stage 6 \
  --stop-stage 6 \
  --start 0 \
  --stop 10
```

这里的 `start/stop` 是左闭右开区间，用来控制当前 worker 实际处理哪些 split。

## 7. Stage 7-9：MUSAN、BPE、phone

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
  --stage 7 \
  --stop-stage 7
```

如果你使用外部 `--data-root`，这两步都会跟随该根目录：
- stage 2 从 `<data-root>/manifests` 管理 MUSAN manifests
- stage 7 从 `<data-root>/manifests` 读取输入，并把特征写到 `<data-root>/fbank`

### 7.2 标准 BPE

训练前通常需要跑：

```bash
bash prepare.sh \
  --stage 8 \
  --stop-stage 8
```

如果你已经在第 4.2 节里直接复用了共享的 `data/lang_bpe_500`，这一段可以跳过。

### 7.3 可选 phone lexicon

如果你还需要 phone based lang：

```bash
bash prepare.sh \
  --stage 9 \
  --stop-stage 9
```

## 8. 预期产物

核心输出：
- `data/manifests/gigaspeech_{recordings,supervisions}_{M,DEV,TEST}.jsonl.gz`
- `data/fbank/gigaspeech_cuts_{M,DEV,TEST}.jsonl.gz`
- `data/fbank/gigaspeech_feats_{M,DEV,TEST}.lca`
- `data/fbank/gigaspeech_M_split/`
- `data/lang_bpe_500/`

这次稳定跑通 `stage 4` 之后，`M` 子集的关键产物大致是：
- `data/fbank/gigaspeech_feats_M.lca` 约 `111G`
- `data/fbank/gigaspeech_cuts_M.jsonl.gz` 约 `2.1G`

## 9. 训练

如果你现在只想把 `16k` 独立拉起来，优先直接用 [run_train_offline.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/run_train_offline.sh)：

```bash
bash run_train_offline.sh
```

它默认会：
- 使用 `4` 张卡：`0,1,2,3`
- 把实验写到 `../experiments/gigaspeech_16k_train`
- 默认 `WANDB_MODE=offline`
- 在真正启动训练前检查 `wandb` 和 `80` 维特征契约

如果之后你要和别的训练并跑，再显式覆盖 GPU 和端口：

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
  --exp-dir "${EXP_ROOT}/16k"
```

## 10. 解码

解码示例：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/16k" \
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
num_features = 80
```

## 12. 备注

- `stage 5-6` 只是可选辅助链路，不属于标准 `M` 训练主流程
- datamodule 默认把 `--enable-musan` 设为 `False`
- 如果显式传 `--data-root`，MUSAN manifests/features 也会一起落到该根目录下
- 在 CPU-only 机器上做数据准备时，请使用 `--cpu-only true`
- 如果本地 `k2` 构建依赖 CUDA 库，避免在 CPU-only 机器上执行 `train.py` 或 `decode.py --help`
- 在可联网的实例上，可以通过 `bash sync_wandb_offline.sh` 把离线 W&B 记录上传

## 13. 本机 Smoke 与 H200 迁移

这部分只记录已经在本机实际验证过的训练启动行为，以及可以直接迁移到 `8 x H200` 的命令。

### 13.1 本机已验证的事实

- 本机：`2 x 48 GiB` GPU
- 目标训练机：`8 x H200`，单卡约 `141 GiB`
- `2 GPU + FP16 + DDP` 已确认能正常初始化并进入实际训练
- `--num-workers 1` 在本机会触发 DataLoader bus error，本质上是 `/dev/shm` 不足，不是模型 OOM
- 当前本地 smoke 包装脚本默认使用 `--num-workers 0`
- 不要再用 `--print-diagnostics true` 代替 smoke。当前 `FP16 + diagnostics` 会在诊断收尾阶段因为 `torch.linalg_eigh/eig` 的 half precision 限制失败，这不代表训练本身起不来

### 13.2 已实测通过的 16k 双卡 smoke

推荐直接用 [run_smoke_train_offline.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/run_smoke_train_offline.sh)：

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

本次实际结果：

- `max-duration=700`：`peak_reserved_gib_max=17.896`，`avg_step_time_sec_mean=1.048`
- `max-duration=1600`：`peak_reserved_gib_max=35.188`，`avg_step_time_sec_mean=1.778`
- `max-duration=2000`：`peak_reserved_gib_max=43.498`，`avg_step_time_sec_mean=1.903`

结论：

- `2000` 在本机 `48 GiB` 卡上已经完整通过，且显存占用接近本机可用上限
- 考虑到目标训练机是 `141 GiB` H200，且共享内存约 `1000 GiB`，如果你的部署方式是同一台 `8 x H200` 上同时跑 `16k` 和 `24k` 两个任务、每个任务各占 `4` 张卡，那么 `16k` 这个单任务首次正式启动建议直接使用 `WORLD_SIZE=4`、`MAX_DURATION=4000`、`--num-workers 4`

如果你想根据 smoke summary 自动生成建议，可以运行：

```bash
python /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_tools/recommend_h200_config.py \
  /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_smoke/20260405-2x48g/16k_md2000_nw0/zipformer_m_g0-1/smoke_summary.json
```

脚本现在会同时输出：

- 按显存线性缩放得到的 uncapped 建议
- 直接可用的 `production_default`
- 更激进但仍保守的 `production_balanced`

### 13.3 直接迁移到 8 x H200 的命令

这里按你的实际部署方式给命令模板，默认训练机目录结构与本机一致，实验产物统一写到 `/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/`：

- 同一台 `8 x H200` 上并行跑两个任务
- `16k` 占 `GPU 0,1,2,3`
- `24k` 占 `GPU 4,5,6,7`
- 两个任务各自独立设置 `WORLD_SIZE=4`
- 两个任务必须使用不同的 `MASTER_PORT` 和 `EXP_ROOT`

H200 上的 `16k` 四卡 smoke：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WORLD_SIZE=4 \
MASTER_PORT=12354 \
NUM_EPOCHS=1 \
USE_FP16=1 \
USE_WANDB=False \
TENSORBOARD=False \
DATA_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data \
EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/16k_smoke_g0-3 \
SMOKE_NUM_BATCHES=8 \
SMOKE_SKIP_VALIDATION=True \
MAX_DURATION=4000 \
bash launch_train_h200_offline.sh \
  --small-dev true \
  --num-workers 4
```

H200 上的 `16k` 四卡正式训练建议：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WORLD_SIZE=4 \
MASTER_PORT=12354 \
NUM_EPOCHS=30 \
USE_FP16=1 \
MAX_DURATION=4000 \
DATA_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data \
EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/16k_train_g0-3 \
WANDB_MODE=offline \
bash launch_train_h200_offline.sh \
  --num-workers 4
```

如果 `16k` 这个四卡 smoke 在 `MAX_DURATION=4000` 下仍然有明显余量，再继续向 `4400-4800` 提升；如果首发就出现不稳定，再先把 `MAX_DURATION` 下调到 `3200`，不要先回退到 `2000`。

如果你要和 `24k` 同机同时启动，`16k` 这边保持上面的 `GPU 0-3 / MASTER_PORT=12354`，`24k` 侧改用 `GPU 4-7 / MASTER_PORT=12364` 即可，两个任务不要共用同一个 `EXP_ROOT`。

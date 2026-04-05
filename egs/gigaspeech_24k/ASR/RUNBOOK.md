# GigaSpeech 24k 运行手册

本文档说明 `egs/gigaspeech_24k/ASR` 这个 recipe 的推荐执行流程。

这个 recipe 的核心目标是在 GigaSpeech ASR 流水线中使用 24 kHz F5-TTS 风格 log-mel 特征：

- 原始音频主要仍然是 16 kHz
- 先把 recordings 离线升采样到 24 kHz
- 再基于 24 kHz recordings 生成 raw cuts 和 100 维特征
- 训练、解码和导出入口统一消费这套 100 维特征

数据准备可以在 CPU-only 机器上完成。
训练和解码建议在 GPU 机器上完成。

## 1. 进入目录

从 Icefall 仓库根目录进入：

```bash
cd egs/gigaspeech_24k/ASR
```

## 2. 环境准备

激活你平时用于 Icefall 的环境。如果你的本地环境需要，也请确保仓库根目录已经加入 `PYTHONPATH`。

例如：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall:${PYTHONPATH}
```

如果要做统一跟踪，可以设置这些可选环境变量：

```bash
export WANDB_PROJECT=gigaspeech-f5tts-vs-fbank
export WANDB_GROUP=gsm-compare-20260403
export EXP_ROOT=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gsm_compare_20260403
mkdir -p "${EXP_ROOT}"
```

如果不使用 W&B，后面的命令里去掉 `--use-wandb` 相关参数即可。

## 3. 可选：环境自检

如果当前机器是 CPU-only 机器，不要运行 `train.py` 或 `decode.py --help`，因为 GPU 版本的 `k2` 可能会在导入时尝试加载 CUDA 相关库。

这类检查建议只在真正的 GPU 训练机上进行。

## 4. 准备 manifests

生成 `M`、`DEV` 和 `TEST` 的 manifests：

```bash
bash prepare.sh --stage 1 --stop-stage 1 --cpu-only true
```

## 5. Stage 编号速查

这个 recipe 的主流程现在是：

| Stage | 作用 |
| --- | --- |
| 1 | 准备 GigaSpeech manifests |
| 3 | 切分 `M` 的 recordings |
| 4 | 离线升采样 DEV / TEST / `M` recordings |
| 5 | 文本预处理并基于 recordings 生成 raw cuts |
| 6 | 计算 DEV / TEST / `M` 的主特征 |
| 7 | 把 `M` raw cuts 切成特征分片 |
| 8 | 计算 `M` 分片特征 |
| 10 | 准备 BPE |
| 11 | 准备 phone lexicon |

默认 `prepare.sh` 会跑到 stage 6。

## 6. 离线升采样 recordings

先运行 stage 3，把 `M` subset 的 recordings 分成 shards：

```bash
bash prepare.sh --stage 3 --stop-stage 3 --cpu-only true
```

然后运行 stage 4，对 DEV / TEST 和 `M` 的 recording shards 做离线升采样：

```bash
bash prepare.sh \
  --stage 4 \
  --stop-stage 4 \
  --cpu-only true \
  --resample-num-workers 24
```

单机情况下，stage 4 的主要瓶颈通常是 CPU、磁盘写入和音频解码吞吐。

如果你要多实例并行，可以直接使用 helper：

```bash
./run_resample_shard.sh --instance-index 0
./run_resample_shard.sh --instance-index 1
./run_resample_shard.sh --instance-index 2
./run_resample_shard.sh --instance-index 3
```

默认产物路径：

- `data/manifests_resampled/24000/gigaspeech_recordings_DEV.jsonl.gz`
- `data/manifests_resampled/24000/gigaspeech_recordings_TEST.jsonl.gz`
- `data/manifests_resampled/24000/recordings_M_split_1000/`
- `data/audio_cache/gigaspeech/24000/`

## 7. 预处理并计算 24k 特征

stage 5 会读取原始 supervisions，但在 `use_resampled_audio=true` 时改用 stage 4 生成的 24 kHz recordings 来构建 raw cuts。

```bash
bash prepare.sh --stage 5 --stop-stage 5 --cpu-only true
```

stage 6 再基于这些 raw cuts 计算 100 维 F5-TTS log-mel 特征：

```bash
CUDA_VISIBLE_DEVICES='' bash prepare.sh \
  --stage 6 \
  --stop-stage 6 \
  --cpu-only true \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

也可以一次跑完 stage 5-6：

```bash
CUDA_VISIBLE_DEVICES='' bash prepare.sh \
  --stage 5 \
  --stop-stage 6 \
  --cpu-only true \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

参数设计原则：

- `feature-batch-duration` 决定每批总音频秒数，是影响峰值内存的主开关，优先调它。
- `feature-num-workers` 影响音频读取和 DataLoader 预取；收益通常小于大 batch 带来的风险。
- 如果首要目标是稳定，优先把 `feature-num-workers` 控制在 `2-4`。
- 如果当前机器是 CPU-only 机器，建议显式清空 `CUDA_VISIBLE_DEVICES`，避免错误地落到某张卡上。

监控建议：

```bash
watch -n 10 free -h
watch -n 10 'cat /sys/fs/cgroup/memory.current; echo; cat /sys/fs/cgroup/memory.events'
```

## 8. 可选：split-based M 特征计算

如果你要把 `M` 的特征计算进一步拆分，可以使用 stage 7-8：

```bash
bash prepare.sh \
  --stage 7 \
  --stop-stage 8 \
  --cpu-only true \
  --feature-num-splits 100 \
  --feature-start 0 \
  --feature-stop 25 \
  --feature-num-workers 4 \
  --feature-batch-duration 200
```

这不是主 `M` 训练路径的必需步骤。

## 9. 准备 BPE

运行 stage 10：

```bash
bash prepare.sh --stage 10 --stop-stage 10 --cpu-only true
```

## 10. 检查特征维度

确认预计算特征是 100 维：

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

## 11. 训练 Zipformer

切换到 GPU 训练机后再执行这一步。

如果 `24k` 特征还没有准备好，就先不要启动训练；本目录的 helper 会在缺少 `cuts_M` / `lang_bpe_500` 或特征维度不是 `100` 时直接退出。

当 `24k` 数据准备完成后，可以把它当成一个独立训练直接启动：

```bash
bash run_train_offline.sh
```

这个脚本默认会：

- 使用 `4` 张卡：`0,1,2,3`
- 把实验写到 `../experiments/gigaspeech_24k_train`
- 以 `WANDB_MODE=offline` 启动，并把离线记录写到 `${EXP_ROOT}/wandb_offline`
- 在真正启动训练前检查 `wandb` 和 `100` 维特征契约

如果你之后要和其他训练并跑，再显式覆盖 GPU 和端口：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=12364 bash run_train_offline.sh
```

多卡训练示例：

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

## 12. 解码

解码示例：

```bash
python zipformer/decode.py \
  --epoch 30 \
  --avg 15 \
  --exp-dir "${EXP_ROOT}/24k" \
  --max-duration 600 \
  --decoding-method modified_beam_search
```

## 13. 预期输出

重要输出包括：

- `data/manifests_resampled/24000/`
- `data/audio_cache/gigaspeech/24000/`
- `data/fbank/gigaspeech_cuts_DEV.jsonl.gz`
- `data/fbank/gigaspeech_cuts_M.jsonl.gz`
- `data/lang_bpe_500/`
- `${EXP_ROOT}/24k/tensorboard/`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-dev-*.txt`
- `${EXP_ROOT}/24k/modified_beam_search/wer-summary-test-*.txt`
- `${EXP_ROOT}/24k/wandb_run_id.txt`，当启用 W&B 时会生成
- `${EXP_ROOT}/wandb_offline/`，当以离线模式记录 W&B 时会生成

## 14. 备注

- 主 Zipformer 路径默认把 `--enable-musan` 设为 `False`。
- 在 CPU-only 机器上做预处理时，请使用 `--cpu-only true`。
- 如果你正在从旧的“在线重采样”产物切换到新的离线升采样流程，建议先清理旧的 `data/fbank/gigaspeech_cuts_*.jsonl.gz` 和对应 feature 存储后再重跑。
- 如果你正在和 `gigaspeech_16k` 做并行对比，建议使用同一个 W&B project/group，但把 `exp-dir` 分到不同子目录。
- 在可联网的实例上，可以通过 `bash sync_wandb_offline.sh` 把 `${EXP_ROOT}/wandb_offline` 里的离线 run 上传到 W&B。

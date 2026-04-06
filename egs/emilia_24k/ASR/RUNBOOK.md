# Emilia 24k ASR 运行手册

本运行手册对应提交 `9fce537a`。

标准主流程脚本：`prepare.sh`。

本 recipe 的特征配置：
- 音频目标采样率：24 kHz
- 默认启用离线音频缓存
- 声学特征：兼容 F5-TTS 的 mel
- 训练输入特征维度：100

采样率注意点：
- `emilia_24k` 的默认原则是 `raw source -> 24 kHz` 单次重采样
- 不推荐走 `raw source -> intermediate sample rate -> 24 kHz` 的级联重采样
- 默认 stage 3 会直接从原始 source 生成 `24 kHz` FLAC 缓存
- 默认 stage 4 会优先使用这些 resampled manifests 生成 raw cuts
- 默认批量 `zipformer/decode.py` 消费的是预计算好的 `24 kHz` 特征
- 如果后续补在线取波形入口，也应继续保持 `raw source -> 24 kHz` 单次重采样

`prepare_data.sh` 只是兼容性包装脚本，内部会直接转发到 `prepare.sh`。

## 1. 环境

所有命令都在下面目录执行：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR
```

推荐把常用环境变量合并成下面这一段，直接复制即可：

```bash
cd /inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR
source /opt/conda/etc/profile.d/conda.sh
conda activate icefall
export PYTHONPATH=/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/opt/conda/envs/icefall/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/envs/icefall/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PUBLIC_ROOT=/inspire/hdd/project/embodied-multimodality/public
export DATASET_ROOT=/inspire/dataset/emilia/fc71e07
export ARTIFACT_ROOT=$PUBLIC_ROOT/emilia/fc71e07/icefall_emilia_zh_24k
export EMILIA_ARTIFACT_ROOT=$ARTIFACT_ROOT
export LANG=zh
```

代码里的默认数据集根目录已经切到 `/inspire/dataset/emilia/fc71e07`。根据 [fc71e07_dataset_report.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/fc71e07_dataset_report.md)，这就是当前本地 Emilia 副本的真实根目录。这个 recipe 的默认输出根目录也已经切到当前项目的 `public`，也就是 `/inspire/hdd/project/embodied-multimodality/public`。推荐的 `zh` 工作目录布局如下：

```bash
PUBLIC_ROOT=/inspire/hdd/project/embodied-multimodality/public
DATASET_ROOT=/inspire/dataset/emilia/fc71e07
ARTIFACT_ROOT=$PUBLIC_ROOT/emilia/fc71e07/icefall_emilia_zh_24k
EMILIA_ARTIFACT_ROOT=$ARTIFACT_ROOT
LANG=zh
```

在这个布局下，`prepare.sh`、`run_public_resample_shard.sh`、`zipformer/train.py`、`zipformer/decode.py`、`zipformer/export.py` 的默认输入/输出路径都已经对齐到这组路径；不传对应参数时就会直接使用这里的默认值。

在这个布局下：
- 原始 32 kHz 音频保留在 `$DATASET_ROOT/ZH`
- 重采样后的 24 kHz FLAC 缓存写入 `$ARTIFACT_ROOT/audio_cache`
- manifests、cuts、MUSAN 相关产物和 BPE 相关产物写入 `$ARTIFACT_ROOT/data`
- 训练输出也可以统一写入 `$ARTIFACT_ROOT/exp`

这套布局的关键点是：
- 原始音频和 `24 kHz` 缓存明确分层保存
- 训练和默认批量解码都围绕 `24 kHz` 特征产物展开
- 避免把原始音频先降到别的中间采样率，再升到 `24 kHz`

报告里还确认了当前本地副本的关键信息：
- 根目录包含 `DE/ EN/ FR/ JA/ KO/ ZH/` 六种语言
- `ZH/` 共有 92 个顶层 batch/jsonl 分片
- 中文 JSONL 总条目数约 19,969,297
- `wav` 字段是相对于 `ZH/` 或 `EN/` 的相对路径

## 2. Stage 对照表

`prepare.sh` 的阶段划分如下：

1. Stage 0：生成 Lhotse manifests
2. Stage 1：把 train recordings 切分成 recording shards
3. Stage 2：可选的 MUSAN manifest 准备
4. Stage 3：离线重采样 recordings 到 24 kHz FLAC 缓存
5. Stage 4：文本归一化并生成 raw cuts
6. Stage 5：计算 dev/test 的 F5 mel 特征
7. Stage 6：切分 train raw cuts
8. Stage 7：按分片计算 train 的 F5 mel 特征
9. Stage 8：可选的 MUSAN 特征
10. Stage 9：合并 train split cut manifests
11. Stage 10：准备 BPE 语言目录

重要默认参数：
- `target_sample_rate=24000`
- `use_resampled_audio=true`
- `speed_perturb=true`
- `enable_musan=false`
- `recording_num_splits=1000`
- `num_splits=100`
- `num_workers=20`
- `batch_duration=600`
- `feature_device=auto`

`local/compute_fbank_emilia.py` 使用的特征提取器是 `local/f5tts_mel_extractor.py`，训练特征维度在 `zipformer/train.py` 中固定为 100。

## 3. 按 Stage 与设备划分的运行方式

先按用途把 stage 分清楚，再按 CPU/GPU 选命令：

| Stage | 内容 | 推荐设备 | 并行方式 |
|---|---|---|---|
| 0 | 生成 manifests | CPU | 单机一次 |
| 1 | 切分 train recordings | CPU | 单机一次 |
| 2 | 准备 MUSAN manifests | CPU | 单机一次，可选 |
| 3 | 32k -> 24k 离线重采样 | CPU | 多实例分片 |
| 4 | 文本归一化 + raw cuts | CPU | 单机一次 |
| 5 | dev/test 特征提取 | CPU 或 GPU | 单机一次 |
| 6 | 切分 train raw cuts | CPU | 单机一次 |
| 7 | train 特征提取 | CPU 或 GPU | 多实例/多卡分片 |
| 8 | MUSAN 特征 | CPU | 单机一次，可选 |
| 9 | 合并 train cut manifests | CPU | 单机一次 |
| 10 | 准备 BPE 目录 | CPU | 单机一次 |

默认建议：
- Stage 0-4：全部按 CPU 跑
- Stage 5：优先按 CPU 跑，简单稳定
- Stage 6：按 CPU 跑
- Stage 7：数据量大，按 CPU 多实例分片或 GPU 多卡分片二选一
- Stage 8-10：按 CPU 跑
- 默认批量 `zipformer/decode.py` 消费的是预计算好的 `24 kHz` 特征
- 当前目录没有 recipe-local 的 `streaming_decode.py`；如果后续补上，仍应直接从原始 source 一次重采样到 `24 kHz`

## 4. Stage 0-4：数据准备与离线重采样

这一段全部是 CPU 流程。

### 4.1 最简单的单机串行跑法

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 0 \
  --stop-stage 4
```

### 4.2 Stage 3：多实例 CPU 离线重采样

Stage 3 是 CPU 导向的，只要满足下面条件，就可以安全地在多个 CPU 实例之间分片运行：
- 所有实例都指向同一个 `DATASET_ROOT`
- 所有实例都指向同一个 `ARTIFACT_ROOT`
- 每个实例使用互不重叠的 shard 区间
- 不要传 `--overwrite`

先由一个协调实例只跑 `stage 0-1`：

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --recording-num-splits 1000 \
  --stage 0 \
  --stop-stage 1
```

然后启动多实例重采样。这个 recipe 提供了 [run_public_resample_shard.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/run_public_resample_shard.sh)，支持 `--detach true`，会自动用 `nohup` 后台运行并打印 PID、launcher 日志和 worker 日志。

说明：
- 这里的 stage 3 离线缓存是直接按原始 source 解码后一次重采样到 `24 kHz`
- 默认目标是把后续 raw cuts、特征提取和解码都统一到这套 `24 kHz` 产物上
- 已经完成的 shard 会按输出 manifest 自动复用；重新启动时不要删除已完成结果，也不要传 `--overwrite`
- 现在 stage 3 增加了共享盘 shard 锁；即使多个机器误碰到同一个 shard，也会只有一个 worker 真正处理它，其他 worker 会跳过

4 实例示例：

```bash
./run_public_resample_shard.sh --instance-index 0 --detach true
./run_public_resample_shard.sh --instance-index 1 --detach true
./run_public_resample_shard.sh --instance-index 2 --detach true
./run_public_resample_shard.sh --instance-index 3 --detach true
```

在 `recording_num_splits=1000`、`num_instances=4` 时，区间是：
- worker 0：`[0, 250)`
- worker 1：`[250, 500)`
- worker 2：`[500, 750)`
- worker 3：`[750, 1000)`

只有 worker 0 会顺带处理 `dev/test` 的重采样。

每个实例会生成两类日志：
- launcher 日志：`$ARTIFACT_ROOT/logs/launcher.resample.<lang>.<instance>of<num_instances>.nohup.log`
- pid 文件：`$ARTIFACT_ROOT/logs/launcher.resample.<lang>.<instance>of<num_instances>.pid`
- 实际 worker 日志：`$ARTIFACT_ROOT/logs/resample.<lang>.<start>-<stop>.log`

### 4.3 Stage 3：按不均匀剩余 shard 重启

如果 stage 3 已经跑了一段时间，完成分布明显不均匀，就不要再用均匀切片方式重启。当前这套 fc71e07 中文数据的实际剩余 shard 可以整理成下面 8 段：

- 机器 0：`[92, 171)`
- 机器 1：`[171, 250)`
- 机器 2：`[371, 435)`
- 机器 3：`[435, 500)`
- 机器 4：`[610, 680)`
- 机器 5：`[680, 750)`
- 机器 6：`[843, 921)`
- 机器 7：`[921, 1000)`

这时不要再手写 `nohup bash prepare.sh ... --resample-start ... --resample-stop ... &`。请直接使用 [run_public_resample_range.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/run_public_resample_range.sh)：

- 它固定执行 `stage 3`
- 它按 `resample-start/resample-stop` 启动，不再按 `instance-index` 均分
- 它自带启动锁、pid 文件和同 range 查重，避免同一台机器重复起两个 `prepare.sh`

这个阶段配套的停机脚本仍然是 [stop_public_resample_shard.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/stop_public_resample_shard.sh)。现在它也支持直接传 `--resample-start/--resample-stop`，可以按真实 range 杀整棵 stage 3 进程树，并补抓对应 shard 的孤儿 `resample_recordings_to_flac.py`。

重启前的原则：

- 先在每台机器上停掉它本机已有的旧 stage 3 进程
- 对于机器 0-3，如果之前跑过旧的 4 机大区间，还要先把对应旧大区间停掉
- 对于所有机器，都再停一次本机将要接管的新 range，确保本机没有重复任务
- 停完后用 `pgrep` 确认本机没有残留 stage 3 进程，再执行新的 range launcher

执行清单如下。

机器 0：

```bash
./stop_public_resample_shard.sh --resample-start 0 --resample-stop 250
./stop_public_resample_shard.sh --resample-start 92 --resample-stop 171
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 92 --resample-stop 171
```

机器 1：

```bash
./stop_public_resample_shard.sh --resample-start 250 --resample-stop 500
./stop_public_resample_shard.sh --resample-start 171 --resample-stop 250
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 171 --resample-stop 250
```

机器 2：

```bash
./stop_public_resample_shard.sh --resample-start 500 --resample-stop 750
./stop_public_resample_shard.sh --resample-start 371 --resample-stop 435
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 371 --resample-stop 435
```

机器 3：

```bash
./stop_public_resample_shard.sh --resample-start 750 --resample-stop 1000
./stop_public_resample_shard.sh --resample-start 435 --resample-stop 500
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 435 --resample-stop 500
```

机器 4：

```bash
./stop_public_resample_shard.sh --resample-start 610 --resample-stop 680
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 610 --resample-stop 680
```

机器 5：

```bash
./stop_public_resample_shard.sh --resample-start 680 --resample-stop 750
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 680 --resample-stop 750
```

机器 6：

```bash
./stop_public_resample_shard.sh --resample-start 843 --resample-stop 921
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 843 --resample-stop 921
```

机器 7：

```bash
./stop_public_resample_shard.sh --resample-start 921 --resample-stop 1000
sleep 3
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
./run_public_resample_range.sh --resample-start 921 --resample-stop 1000
```

如果 `sleep 3` 之后那条 `pgrep` 还有残留，再手动补一轮强杀：

```bash
pkill -9 -f 'prepare.sh .*--stage 3'
pkill -9 -f 'local/resample_recordings_to_flac.py'
sleep 2
pgrep -af 'prepare.sh .*--stage 3|local/resample_recordings_to_flac.py'
```

range launcher 的日志和 pid 文件默认写到：

```bash
$ARTIFACT_ROOT/logs/resample.<lang>.<start>-<stop-1>.rebalance.log
$ARTIFACT_ROOT/logs/launcher.resample.range.<lang>.<start>-<stop-1>.nohup.log
$ARTIFACT_ROOT/logs/launcher.resample.range.<lang>.<start>-<stop-1>.pid
```

共享盘锁仍然落在：

```bash
$ARTIFACT_ROOT/locks/resample/$LANG/24000/recordings_train_split_1000
```

重启后，这套不均匀分片方案会继续复用已经完成的 shard：

- 如果目标 output manifest 已存在，整个 shard 会直接跳过
- 如果某个 shard 正好在迁移瞬间被别的机器抢到，共享盘锁会让后来的 worker 跳过该 shard，避免撞写

完成后继续跑 stage 4：

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 4 \
  --stop-stage 4
```

## 5. Stage 5：dev/test 特征提取

Stage 5 只处理 `dev` 和 `test`，没有 train 分片问题。这里建议单机一次跑完。

### 5.1 CPU 跑法

推荐参数：
- `--feature-device cpu`
- `--feature-num-workers 0`
- `--feature-batch-duration 300`

推荐直接使用 [run_public_cpu_stage5.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/run_public_cpu_stage5.sh)，它固定执行 stage 5，并支持 `--detach true`。

前台运行：

```bash
./run_public_cpu_stage5.sh
```

后台运行：

```bash
./run_public_cpu_stage5.sh --detach true
```

它会生成两类日志：
- launcher 日志：`$ARTIFACT_ROOT/logs/launcher.feature.cpu.stage5.<lang>.nohup.log`
- 实际 worker 日志：`$ARTIFACT_ROOT/logs/feature.cpu.stage5.<lang>.log`

如果你仍然想手动调用 `prepare.sh`，命令如下：

```bash
CUDA_VISIBLE_DEVICES="" \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 5 \
  --stop-stage 5 \
  --feature-device cpu \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

### 5.2 GPU 跑法

如果机器上有 GPU，也可以让 Stage 5 自动用 GPU：

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 5 \
  --stop-stage 5 \
  --feature-device auto \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

## 6. Stage 6-7：train 特征提取

Stage 6 先切 train raw cuts，Stage 7 再真正提取 train 特征。

### 6.1 Stage 6：先切分 train raw cuts

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --feature-num-splits 100 \
  --stage 6 \
  --stop-stage 6
```

### 6.2 Stage 7：CPU 多实例分片跑法

Stage 7 可以完全在 CPU 上跑。这个 recipe 提供了 [run_public_cpu_feature_shard.sh](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/run_public_cpu_feature_shard.sh)，固定执行 stage 7，并自动根据 `--instance-index` 和 `--num-instances` 计算 `feature_start/feature_stop`。它也支持 `--detach true`，会自动用 `nohup` 后台运行。

4 实例示例：

```bash
./run_public_cpu_feature_shard.sh --instance-index 0 --detach true
./run_public_cpu_feature_shard.sh --instance-index 1 --detach true
./run_public_cpu_feature_shard.sh --instance-index 2 --detach true
./run_public_cpu_feature_shard.sh --instance-index 3 --detach true
```

如果 `feature_num_splits=100` 且 `num_instances=4`，区间是：
- worker 0：`[0, 25)`
- worker 1：`[25, 50)`
- worker 2：`[50, 75)`
- worker 3：`[75, 100)`

每个实例会生成两类日志：
- launcher 日志：`$ARTIFACT_ROOT/logs/launcher.feature.cpu.<lang>.<instance>.nohup.log`
- 实际 worker 日志：`$ARTIFACT_ROOT/logs/feature.cpu.<lang>.<start>-<stop>.log`

### 6.3 Stage 7：GPU 多卡分片跑法

如果要用 GPU 跑 Stage 7，建议手动按 `feature_start/feature_stop` 分片：

Worker 0：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 7 \
  --stop-stage 7 \
  --feature-start 0 \
  --feature-stop 25 \
  --feature-device auto \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

Worker 1：

```bash
CUDA_VISIBLE_DEVICES=1 \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 7 \
  --stop-stage 7 \
  --feature-start 25 \
  --feature-stop 50 \
  --feature-device auto \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

按同样方式继续覆盖全部 `train_split_<N>` shards。

### 6.4 Stage 4-10 一口气纯 CPU 跑完

如果你想在 CPU 机器上从 raw cuts 一路跑到 train 特征结束：

```bash
CUDA_VISIBLE_DEVICES="" \
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 4 \
  --stop-stage 10 \
  --feature-device cpu \
  --feature-num-workers 0 \
  --feature-batch-duration 300
```

## 7. Stage 8-10：MUSAN、合并 cuts、BPE

这一段全部是 CPU 流程。

### 7.1 可选 MUSAN

如果需要 MUSAN：

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --enable-musan true \
  --stage 2 \
  --stop-stage 2
```

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --enable-musan true \
  --stage 8 \
  --stop-stage 8
```

这条链路现在和主产物目录完全对齐：
- stage 2 会把 MUSAN manifests 写到 `$ARTIFACT_ROOT/data/manifests`
- stage 8 会从 `$ARTIFACT_ROOT/data/manifests` 读取输入，并把特征写到 `$ARTIFACT_ROOT/data/fbank/$LANG`

如果保持默认的 `enable_musan=false`，则训练阶段不需要再额外加参数，因为这个 recipe 的 datamodule 默认也是 `--enable-musan false`。

### 7.2 标准 CPU 收尾命令

如果 Stage 8 不需要 MUSAN，常见的收尾方式是直接继续跑 `stage 9-10`：

```bash
bash prepare.sh \
  --language "$LANG" \
  --dataset-root "$DATASET_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --stage 9 \
  --stop-stage 10
```

## 8. 预期产物

核心输出：
- `$ARTIFACT_ROOT/data/manifests/<lang>/emilia_<lang>_recordings_{train,dev,test}.jsonl.gz`
- `$ARTIFACT_ROOT/data/manifests_resampled/<lang>/24000/...`
- `$ARTIFACT_ROOT/data/fbank/<lang>/emilia_<lang>_cuts_{dev,test,train}.jsonl.gz`
- `$ARTIFACT_ROOT/data/fbank/<lang>/train_split_<N>/emilia_<lang>_cuts_train.*.jsonl.gz`
- `$ARTIFACT_ROOT/data/lang_bpe_<lang>_<vocab_size>/`

训练时可以读取：
- 合并后的 train cuts：`$ARTIFACT_ROOT/data/fbank/<lang>/emilia_<lang>_cuts_train.jsonl.gz`
- 或按分片保存的 train cuts：`$ARTIFACT_ROOT/data/fbank/<lang>/train_split_*`

## 9. 训练

默认实验目录：
- zh：`$ARTIFACT_ROOT/exp/zipformer/exp-zh-24k`
- en：`$ARTIFACT_ROOT/exp/zipformer/exp-en-24k`

推荐的 8 卡训练方式，TensorBoard 和 W&B 写到同一个共享项目：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

```bash
python zipformer/train.py \
  --world-size 8 \
  --language "$LANG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 1000 \
  --tensorboard true \
  --use-wandb true \
  --wandb-project emilia-asr \
  --wandb-group "${LANG}-compare" \
  --wandb-run-name "emilia-${LANG}-24k-f5tts" \
  --wandb-tags "emilia,24k,f5tts-mel"
```

TensorBoard 日志写入：

```text
$ARTIFACT_ROOT/exp/zipformer/exp-<lang>-24k/tensorboard
```

恢复训练：

```bash
python zipformer/train.py \
  --world-size 8 \
  --language "$LANG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --start-epoch 11 \
  --use-fp16 1 \
  --max-duration 1000 \
  --use-wandb true \
  --wandb-project emilia-asr \
  --wandb-group "${LANG}-compare" \
  --wandb-run-name "emilia-${LANG}-24k-f5tts"
```

## 10. 解码

解码示例：

```bash
python zipformer/decode.py \
  --language "$LANG" \
  --epoch 30 \
  --avg 15 \
  --exp-dir "$ARTIFACT_ROOT/exp/zipformer/exp-${LANG}-24k" \
  --max-duration 600 \
  --decoding-method greedy_search
```

## 11. 快速检查

检查特征 manifests：

```bash
python - <<'PY'
from lhotse import load_manifest_lazy
from os import environ
artifact_root = environ["ARTIFACT_ROOT"]
for split in ["dev", "test", "train"]:
    p = f"{artifact_root}/data/fbank/zh/emilia_zh_cuts_{split}.jsonl.gz"
    try:
        cuts = load_manifest_lazy(p)
        first = next(iter(cuts))
        print(split, first.num_features)
    except Exception as e:
        print(split, e)
PY
```

预期特征维度是 `100`。

检查训练 CLI：

```bash
python zipformer/train.py --help | rg 'wandb|tensorboard|language|exp-dir'
```

检查解码 CLI：

```bash
python zipformer/decode.py --help >/tmp/emilia_24k_decode_help.txt
```

## 12. 备注

- Stage 3 会写入缓存后的 24 kHz FLAC 文件以及更新后的 manifests。后续特征提取会通过 Lhotse 直接读取这些缓存音频，而不是再做一层手动 WAV 转换。
- F5 风格特征提取器本身也可以内部重采样到 24 kHz，但这个 recipe 的设计目标是通过离线 24 kHz 缓存来提升大规模数据处理时的可扩展性。
- Stage 9 建议执行，但不是训练的硬性前置条件，因为 datamodule 在必要时可以直接回退到 split train manifests。

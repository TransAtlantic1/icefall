# GigaSpeech 24k 审查记录

## 审查范围

- 目录：`egs/gigaspeech_24k`
- 重点：采样率配置、离线重采样、特征计算方式、并行算特征、训练流程，以及“文件功能和文件命名是否一致”
- 实际审查重点集中在 `ASR/`，因为主要改动都在这里；`KWS/` 基本还是从 16k 配方平移过来的结构

## 主要结论

### P0: `zipformer` 的部分入口仍然写死 16k/80 维，和当前 24k/100 维训练配置不一致

- [`ASR/zipformer/streaming_decode.py`](./ASR/zipformer/streaming_decode.py) 第 554-555 行仍然固定使用 `16000` 和 `80` 维 fbank。
- [`ASR/zipformer/export-onnx.py`](./ASR/zipformer/export-onnx.py) 第 297 行仍然用 `(1, 100, 80)` 作为导出 trace 输入。
- 但 [`ASR/zipformer/train.py`](./ASR/zipformer/train.py) 已经把配方配置成 `feature_type=f5tts_mel`、`feature_sample_rate=24000`、`feature_dim=100`。

影响：

- 流式解码入口如果直接复用，会和训练时的 24k F5-TTS mel 前端不一致。
- ONNX 导出如果按当前代码执行，输入 shape 约定仍是 80 维，和训练出的 100 维模型不匹配。

建议：

- 如果这两个入口暂时不支持 24k，就在文件头和 README 中明确标注“未迁移，不可用于 24k recipe”。
- 如果要支持 24k，就把前端和输入 shape 全部改成从统一配置读取，而不是局部写死。

### P0: `conformer_ctc` 和 `pruned_transducer_stateless2` 实际上还停留在 80 维 fbank 语义，目录名已经是 24k，但功能并未完成迁移

`conformer_ctc`：

- [`ASR/conformer_ctc/train.py`](./ASR/conformer_ctc/train.py) 第 208 行仍然是 `feature_dim = 80`
- [`ASR/conformer_ctc/decode.py`](./ASR/conformer_ctc/decode.py) 第 159 行仍然是 `feature_dim = 80`
- [`ASR/conformer_ctc/asr_datamodule.py`](./ASR/conformer_ctc/asr_datamodule.py) 第 252、300、327 行在 `on_the_fly_feats` 分支里仍然调用 `Fbank(FbankConfig(num_mel_bins=80))`

`pruned_transducer_stateless2`：

- [`ASR/pruned_transducer_stateless2/train.py`](./ASR/pruned_transducer_stateless2/train.py) 第 339 行仍然是 `feature_dim = 80`
- [`ASR/pruned_transducer_stateless2/asr_datamodule.py`](./ASR/pruned_transducer_stateless2/asr_datamodule.py) 第 285、342、369 行仍然在 `on_the_fly_feats` 分支里生成 80 维 fbank

影响：

- 现在这个目录名会让人默认以为整个 `gigaspeech_24k/ASR` 都完成了 24k 迁移，但实际上只有 Zipformer 主链路比较完整。
- 后续如果有人直接在这个目录里跑 conformer/transducer，很容易遇到特征维度不匹配，或者 silently 走错前端。
- README 的“24k performance record”把这两个模型一起列出来，也会进一步加重误导。

建议：

- 如果当前项目只验证 Zipformer，就把另外两个子目录明确标成“legacy copy / 未迁移 / 不在本次实验范围”。
- 更彻底的做法是把未迁移的子目录暂时移出 `gigaspeech_24k`，或者补齐到真正的 24k/F5-TTS mel 版本。

### P1: 多个文件名和目录名仍然沿用 `fbank`，但真实功能已经变成“24k F5-TTS log-mel”

直接证据：

- [`ASR/local/compute_fbank_gigaspeech.py`](./ASR/local/compute_fbank_gigaspeech.py) 第 75 行函数名仍叫 `compute_fbank_gigaspeech`，第 76 行输出目录仍是 `data/fbank`，但第 90 行实际使用的是 `F5TTSMelExtractor`
- [`ASR/local/compute_fbank_gigaspeech_splits.py`](./ASR/local/compute_fbank_gigaspeech_splits.py) 第 100 行函数名仍叫 `compute_fbank_gigaspeech_splits`，第 102 行输出仍在 `data/fbank/gigaspeech_M_split`，但第 116 行实际也是 `F5TTSMelExtractor`
- [`ASR/prepare.sh`](./ASR/prepare.sh) 第 22 行注释仍写“Compute fbank features”，第 185 行 Stage 4 仍然是“Compute features”，第 214 行甚至还写着“Compute fbank for musan”

影响：

- 对内部维护者来说，“名字像 16k 基线，行为像 24k 新配方”，理解成本很高。
- 对脚本调用者来说，很容易把 `data/fbank` 下的 24k/100 维特征误当成 16k/80 维基线产物。
- 如果后续还要保留 `gigaspeech_16k` 和 `gigaspeech_24k` 并行实验，这种命名会持续制造混淆。

建议：

- 至少把脚本名、函数名、stage 文案改成 `compute_features_*` 或 `compute_f5tts_mel_*`
- `data/fbank` 最好改成更语义化的层级，例如：
  - `data/features/f5tts_mel_24k/`
  - `data/features/kaldifeat_fbank_16k/`
- 这样目录本身就能表达“特征类型 + 采样率”

### P1: 当前 recipe 树里混入了大量运行产物，源代码和实验产物没有分层

当前目录下可见：

- `ASR/core.*`
- `ASR/stage4_24k.log`
- `ASR/data/`
- `ASR/download -> ...` 软链

影响：

- 目录整体现在是 `git status ?? egs/gigaspeech_24k/`，说明一旦误提交，代码、日志、core dump、数据产物会一起进入版本控制。
- 这会让 recipe 目录从“配方源码”变成“源码 + 实验现场快照”的混合体，后续很难维护。

建议：

- recipe 目录只保留代码、README、runbook、少量配置文件。
- `core.*`、`*.log`、`data/`、`download/`、`exp/` 统一放到 repo 外部或 `public` 根目录，并加入 `.gitignore`。

### P2: README 有文档级不一致

- [`ASR/README.md`](./ASR/README.md) 第 32-34 行列出了 `conformer_ctc` 和 `pruned_transducer_stateless2` 的表现，但这两条链路在代码层面并没有完成 24k 迁移。
- [`ASR/README.md`](./ASR/README.md) 第 36 行链接仍然指向 `/egs/gigaspeech/ASR/RESULTS.md`，不是当前 24k recipe 的路径。

建议：

- README 里只保留当前已验证链路的结果。
- 未迁移链路单独标注“copied from 16k, not yet adapted”.

## 我对这个目录的理解

### 1. 这是一个从 `gigaspeech_16k` 平移出来的 24k ASR 配方

核心目标不是重新定义整个 GigaSpeech recipe，而是在现有 icefall GigaSpeech 框架上，替换前端特征：

- 输入音频原始采样率仍然主要是 16k
- 在特征提取阶段做离线重采样到 24k
- 用自定义的 F5-TTS 风格 log-mel 提取器替换原来的 Kaldifeat fbank
- 训练主链路目前主要围绕 `zipformer/` 构建

### 2. 数据准备主链路

- [`ASR/prepare.sh`](./ASR/prepare.sh)
  - Stage 1: 生成 `M/DEV/TEST` manifest
  - Stage 3: `local/preprocess_gigaspeech.py` 把 manifest 转成 raw cuts，并做文本规范化/OOV 过滤
  - Stage 4: `local/compute_fbank_gigaspeech.py` 预计算特征
  - Stage 5/6: 可选，把 `M` 拆分后并行算特征
  - Stage 8: 训练 BPE

### 3. 24k 特征链路

- [`ASR/local/f5tts_mel_extractor.py`](./ASR/local/f5tts_mel_extractor.py)
  - 内部用 `torchaudio` 把 16k 音频重采样到 24k
  - 输出 100 维 mel
- [`ASR/local/compute_fbank_gigaspeech.py`](./ASR/local/compute_fbank_gigaspeech.py)
  - 批量计算 `DEV/TEST/M` 的预计算特征
- [`ASR/local/compute_fbank_gigaspeech_splits.py`](./ASR/local/compute_fbank_gigaspeech_splits.py)
  - 把 `M` 切分后并行计算，提高大规模特征计算吞吐

### 4. 训练主链路

- 当前真正完成“24k 配置闭环”的是 `ASR/zipformer/`
- 它已经把 recipe 标识、feature type、sample rate、feature dim 改到了 24k/100 维
- W&B、runbook、主训练命令也基本围绕这条链路组织

### 5. 当前目录的真实状态

更准确地说，这不是“完整的 24k GigaSpeech recipe”，而是：

- 一个“Zipformer 主链路基本完成迁移”的 24k 实验目录
- 外加若干从 16k 复制过来但尚未完全迁移的子模块

如果项目组要长期维护，这个状态最好在命名和文档上说清楚。

## 对 `public` 目录落盘方式的修改建议

你特别提到“数据处理的结果都放到 public 目录下”，我建议不要再让 recipe 自己默认写相对路径 `data/`，而是把“代码目录”和“产物目录”彻底分离。

### 推荐做法

1. 给 `prepare.sh` 增加三个可配置根目录

- `download_root`
- `artifact_root`
- `exp_root`

例如：

```bash
download_root=${DOWNLOAD_ROOT:-/public/.../gigaspeech/download}
artifact_root=${ARTIFACT_ROOT:-/public/.../gigaspeech_24k}
exp_root=${EXP_ROOT:-/public/.../gigaspeech_24k/exp}
```

然后把下面这些路径都改成基于变量展开：

- `download/`
- `data/manifests/`
- `data/fbank/`
- `zipformer/exp` 或 runbook 里的 `${EXP_ROOT}`

2. 明确区分“共享原始资源”和“实验派生产物”

建议分层：

- `/public/.../gigaspeech_shared/download/`
- `/public/.../gigaspeech_shared/lm/`
- `/public/.../gigaspeech_24k/manifests/`
- `/public/.../gigaspeech_24k/features/f5tts_mel_24k/`
- `/public/.../gigaspeech_24k/exp/zipformer/...`

这样有三个好处：

- 原始下载资源可以多实验共享
- 特征产物和训练产物不会混在 recipe 代码目录
- 目录名自带语义，不会把 16k/24k、fbank/mel 混在一起

3. 如果暂时不想改太多脚本，至少先做软链层

最低成本版本：

- `ASR/data -> /public/.../gigaspeech_24k/data`
- `ASR/download -> /public/.../gigaspeech_shared/download`
- `ASR/zipformer/exp -> /public/.../gigaspeech_24k/exp/zipformer`

这样能先把大体积产物从源码树里挪出去。

4. 把运行产物加入忽略规则

至少忽略：

- `egs/gigaspeech_24k/ASR/data/`
- `egs/gigaspeech_24k/ASR/download`
- `egs/gigaspeech_24k/ASR/*/exp/`
- `egs/gigaspeech_24k/ASR/core.*`
- `egs/gigaspeech_24k/ASR/*.log`

## 建议的后续动作

优先级建议：

1. 先修 `zipformer/streaming_decode.py` 和 `zipformer/export-onnx.py` 的 16k/80 写死问题
2. 再决定 `conformer_ctc`、`pruned_transducer_stateless2` 是继续迁移，还是明确标记为未迁移
3. 再统一 `fbank` 相关脚本和目录命名
4. 最后把 `data/download/exp/log/core` 迁到 `public`，并补 `.gitignore`

# Emilia 24k ASR

这个目录是面向 Emilia 单语种 ASR 的 `24 kHz` recipe。

当前主链路的前端约定是：

- 目标采样率：`24 kHz`
- 声学特征：F5-TTS 风格 log-mel
- 输入维度：`100`
- 默认启用离线重采样缓存

## 采样率原则

`emilia_24k` 统一遵循下面这个原则：

- 优先保持 `raw source -> 24 kHz` 的单次重采样
- 不推荐走 `raw source -> 16/32 kHz intermediate -> 24 kHz` 的级联重采样

默认主流程里，这个原则已经成立：

- `prepare.sh` stage 3 直接从原始 source 离线生成 `24 kHz` FLAC 缓存
- `prepare.sh` stage 4 默认优先使用 stage 3 产出的 resampled manifests 生成 raw cuts
- 默认批量 `zipformer/decode.py` 消费的是预计算好的 `24 kHz` 特征

因此，常规训练和批量解码路径不会退化成多次重采样。

## 注意点

- Emilia 原始音频常见为 `32 kHz`，不要把它理解成“先固定到某个中间采样率，再升到 24k”。
- 如果启用 MUSAN，本地 manifests 和 features 也会跟随 `artifact_root/data` 布局，不再依赖仓库当前目录下的 `data/`。
- 如果后续补充新的在线取波形入口，也应保持 `raw source -> 24 kHz` 的单次重采样原则。
- `zipformer/export.py` 里的示例现在只保留本 recipe 现有的批量 `decode.py` 路径；当前目录没有 recipe-local 的 `streaming_decode.py`。

更完整的执行命令和阶段说明见 [RUNBOOK.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k/ASR/RUNBOOK.md)。

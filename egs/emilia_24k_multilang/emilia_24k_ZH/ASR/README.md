# Emilia 24k ASR

这个目录只针对 `egs/emilia_24k_multilang/emilia_24k_ZH/ASR`。

当前主线配置：

- 目标采样率：`24 kHz`
- 特征：F5-TTS 风格 log-mel
- 输入维度：`100`
- 默认启用离线重采样缓存

主线入口：

- 数据准备：`prepare.sh`
- 训练：`zipformer/train.py`
- 解码：`zipformer/decode.py`
- 导出：`zipformer/export.py`

补充位置：

- 主流程说明见 [RUNBOOK.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/emilia_24k_multilang/emilia_24k_ZH/ASR/RUNBOOK.md)
- 最小真实数据验证脚本见 [test/recipes/emilia24k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/emilia24k)
- 可选 watcher 见 [asr_op/emilia/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/emilia/watcher)

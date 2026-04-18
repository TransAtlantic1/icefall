# GigaSpeech 16k ASR

这个目录是 GigaSpeech `16 kHz` Zipformer recipe，主线目标只有三件事：

- 生成 `80` 维 Kaldifeat fbank 特征
- 训练、解码、导出 `M` 子集模型
- 保持一套适合 CPU 数据准备、GPU 训练的稳定主流程

主线入口：

- 数据准备：`prepare.sh`
- 训练：`run_train_offline.sh`
- 解码：`zipformer/decode.py`
- 导出：`zipformer/export.py`

补充位置：

- 主流程说明见 [RUNBOOK.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/RUNBOOK.md)
- 结果记录模板见 [RESULTS.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/RESULTS.md)
- 最小真实数据验证脚本见 [test/recipes/giga16k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/giga16k)
- 可选 watcher 见 [asr_op/giagspeech/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/giagspeech/watcher)

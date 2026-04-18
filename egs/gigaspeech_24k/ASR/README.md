# GigaSpeech 24k ASR

这个目录是 GigaSpeech `24 kHz` Zipformer recipe，主线目标只有三件事：

- 离线把原始录音一次重采样到 `24 kHz`
- 生成 `100` 维 F5-TTS 风格 log-mel 特征
- 训练、解码、导出 `M` 子集模型

主线入口：

- 数据准备：`prepare.sh`
- 重采样分片：`run_resample_shard.sh`
- 训练：`run_train_offline.sh`
- 解码：`zipformer/decode.py`
- 导出：`zipformer/export.py`

补充位置：

- 主流程说明见 [RUNBOOK.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR/RUNBOOK.md)
- 结果记录模板见 [RESULTS.md](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_24k/ASR/RESULTS.md)
- 最小真实数据验证脚本见 [test/recipes/giga24k](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/test/recipes/giga24k)
- 可选 watcher 见 [asr_op/giagspeech/watcher](/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/asr_op/giagspeech/watcher)

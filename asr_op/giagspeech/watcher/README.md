# GigaSpeech Watcher

这个目录只放 GigaSpeech 相关的可选 watcher 脚本。

- `auto_decode_checkpoints.sh`: 轮询训练输出目录，在目标 checkpoint 出现后自动触发 `zipformer/decode.py`

这些脚本属于运维辅助工具，不属于 recipe 主流程。主线文档只保留到这里的短链接，不再在 `RUNBOOK.md` 中展开机器编排或长期轮询细节。

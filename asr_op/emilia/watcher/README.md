# Emilia Watcher

这个目录只放 Emilia 相关的可选 watcher 脚本。

- `run_auto_decode.sh`: 轮询 Emilia 训练输出目录，对新 checkpoint 自动触发 `zipformer/decode.py`

这些脚本属于运维辅助工具，不属于 recipe 主流程。主线文档只保留到这里的短链接，不再在 `RUNBOOK.md` 中展开长期轮询细节。

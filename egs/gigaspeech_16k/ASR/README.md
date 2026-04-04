# GigaSpeech 16k
GigaSpeech, an evolving, multi-domain English
speech recognition corpus with 10,000 hours of high quality labeled
audio, collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles,
and a variety of topics, such as arts, science, sports, etc. More details can be found: https://github.com/SpeechColab/GigaSpeech

This recipe is a standard Kaldifeat fbank baseline derived from `egs/gigaspeech/ASR`
with 16 kHz experiment metadata and optional W&B logging for train/decode tracking.

## Download

Apply for the download credentials and download the dataset by following https://github.com/SpeechColab/GigaSpeech#download. Then create a symlink
```bash
ln -sfv /path/to/GigaSpeech download/GigaSpeech
```

## Environment

This baseline recipe depends on:

- `torch` and `torchaudio`
- `lhotse`
- `k2`
- `sentencepiece`
- `tensorboard`
- `wandb` when using `--use-wandb True`
- `jq` when regenerating BPE assets

For unified experiment tracking, run both the 16k and 24k recipes with the same W&B `--wandb-project` and `--wandb-group`, and place both `--exp-dir` values under the same parent directory for TensorBoard comparison.

## Performance Record
|                                |  Dev  | Test  |
|--------------------------------|-------|-------|
|           `zipformer`          | 10.25 | 10.38 |
|         `conformer_ctc`        | 10.47 | 10.58 |
| `pruned_transducer_stateless2` | 10.40 | 10.51 |

See [RESULTS](RESULTS.md) for details.

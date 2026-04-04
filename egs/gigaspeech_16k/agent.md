# Review of `egs/gigaspeech_16k`

## Findings

### 1. The recipe still hard-codes processed artifacts under `data/`, which conflicts with the stated project rule that processed outputs should live under `public/`.

- `ASR/prepare.sh:75` says all generated files are saved in `data`, and the script then writes manifests, fbanks, split outputs, and BPE assets under `data/...` throughout `ASR/prepare.sh:158`, `ASR/prepare.sh:171`, `ASR/prepare.sh:177`, `ASR/prepare.sh:194`, `ASR/prepare.sh:215`, and `ASR/prepare.sh:222`.
- `ASR/local/preprocess_gigaspeech.py:71` and `ASR/local/preprocess_gigaspeech.py:72` read from `data/manifests` and write raw cuts to `data/fbank`.
- `ASR/local/compute_fbank_gigaspeech.py:76` writes final cut manifests and feature storage under `data/fbank`.
- `ASR/local/compute_fbank_gigaspeech_splits.py:102` writes split artifacts under `data/fbank/gigaspeech_M_split`.
- Training also defaults to consuming those local artifacts via `ASR/zipformer/asr_datamodule.py:88`.
- The runbook reinforces the same layout with `mkdir -p data/fbank` and `data/lang_bpe_500` in `ASR/RUNBOOK.md:43`, `ASR/RUNBOOK.md:51`, `ASR/RUNBOOK.md:76`, and `ASR/RUNBOOK.md:93`.

Impact: even if the recipe works, it violates the repo-level storage contract you called out, and it makes later sharing/comparison with other recipes harder.

### 2. The recipe is a standard Kaldifeat fbank baseline with 16 kHz metadata and W&B integration, so its naming should explicitly reflect that scope.

- `ASR/local/preprocess_gigaspeech.py:108` through `ASR/local/preprocess_gigaspeech.py:125` only filters text/OOV tags and builds cut manifests; there is no resampling step.
- `ASR/local/compute_fbank_gigaspeech.py:87` through `ASR/local/compute_fbank_gigaspeech.py:113` run standard `KaldifeatFbank` extraction directly from the input cuts; again, no offline resample artifact is created.
- The only explicit 16k marker I found in code is W&B metadata at `ASR/zipformer/train.py:106` through `ASR/zipformer/train.py:110`.
- `ASR/RUNBOOK.md:70` explicitly says the recipe "keeps the standard Kaldifeat fbank extractor".

Impact: after your clarification, this is not a correctness bug. The important follow-up is to keep README, runbook, and code comments aligned with the actual scope: standard Kaldifeat fbank pipeline plus 16 kHz experiment labeling and W&B tracking.

### 3. Several file names, stage descriptions, and help texts no longer match the actual subset handled by the code.

- `ASR/prepare.sh:185` says stage 4 computes features for `DEV, TEST, L, M, S, and XS`, but `ASR/local/compute_fbank_gigaspeech.py:78` through `ASR/local/compute_fbank_gigaspeech.py:85` only process `DEV`, `TEST`, and `M`.
- `ASR/local/compute_fbank_gigaspeech_splits.py:81` says `--num-splits` is "The number of splits of the XL subset", but the script actually operates on `data/fbank/gigaspeech_M_split` at `ASR/local/compute_fbank_gigaspeech_splits.py:102` and reads/writes `gigaspeech_cuts_M...` / `gigaspeech_feats_M...` at `ASR/local/compute_fbank_gigaspeech_splits.py:124`, `ASR/local/compute_fbank_gigaspeech_splits.py:129`, and `ASR/local/compute_fbank_gigaspeech_splits.py:138`.
- `ASR/README.md:1` is still titled plain `# GigaSpeech`, while this directory is specifically a `gigaspeech_16k` derivative.
- `ASR/README.md:36` still links to `/egs/gigaspeech/ASR/RESULTS.md`, not the local 16k recipe results file.

Impact: this is exactly the kind of "file/function/name mismatch" that will confuse future maintenance. The code mostly behaves consistently; the labels and docs do not.

### 4. Generated artifacts and caches are mixed into the recipe tree and are not covered by `.gitignore`.

- `ASR/.gitignore:1` and `ASR/.gitignore:2` ignore only `log-*` and `.DS_Store`.
- The tree currently contains generated content such as `ASR/data/`, `ASR/stage4_16k.log`, `ASR/local/__pycache__/`, and `ASR/zipformer/__pycache__/`.

Impact: because this recipe itself is currently new/untracked, keeping runtime artifacts under the source tree makes the review surface noisy and makes accidental check-in of non-source files more likely.

### 5. The runbook is not portable and still embeds machine-specific paths from another user workspace.

- `ASR/RUNBOOK.md:45` through `ASR/RUNBOOK.md:51` hard-code symlinks to `/inspire/.../junguo/icefall/...`.

Impact: this is not a correctness bug, but it makes the recipe hard to reuse and blurs whether the canonical inputs belong to this recipe or to an external personal workspace.

### 6. The W&B implementation looks structurally sound and matches the neighboring 24k recipe.

- `ASR/zipformer/train.py` initializes W&B only on rank 0, stores a stable run id in `wandb_run_id.txt`, and logs train/valid metrics with explicit global steps.
- `ASR/zipformer/decode.py` resumes the same run id and writes best DEV/TEST decode summaries back into the run summary.
- I did not find a `gigaspeech_16k`-specific W&B bug during review; the implementation is effectively the same pattern used in `egs/gigaspeech_24k`.

## Assumptions and open questions

- I reviewed `egs/gigaspeech_16k` as a new, untracked recipe directory, not as a staged patch against tracked history.
- I assumed your stated rule "data processing outputs should go under `public`" is the target contract for this recipe. Under that assumption, the current path layout is a real issue, not just a style preference.
- This directory intentionally implements the standard Kaldifeat fbank pipeline, with 16 kHz metadata and optional W&B tracking, not a separate offline-resampling pipeline.

## My understanding of this recipe

- `egs/gigaspeech_16k/ASR` is a fork of the standard GigaSpeech ASR recipe intended to serve as a 16 kHz Kaldifeat-fbank baseline for comparison against a 24k recipe.
- The active path is the `zipformer` pipeline, not the inherited `conformer_ctc` or `pruned_transducer_stateless2` baselines.
- The intended flow is:
  1. Prepare manifests for `M`, `DEV`, and `TEST`.
  2. Preprocess transcripts into raw cut manifests.
  3. Compute/store fbank features, mainly for `M`.
  4. Train `zipformer` on subset `M`, with MUSAN disabled by default.
  5. Decode `DEV` and `TEST`.
- Compared with the upstream `egs/gigaspeech/ASR`, the main functional deltas I found are:
  - CPU-only friendly preprocessing entrypoint in `local/preprocess_gigaspeech.py`.
  - Configurable stage-4 feature extraction parallelism in `prepare.sh` and `local/compute_fbank_gigaspeech.py`.
  - `torch.multiprocessing.set_sharing_strategy("file_system")` in the feature scripts.
  - W&B integration in `zipformer/train.py` and `zipformer/decode.py`.
  - `zipformer/asr_datamodule.py` changes the default `--enable-musan` from `True` to `False`.
- The current implementation reads like "a lightly customized fork of the original GigaSpeech fbank recipe for 16 kHz experiment tracking, CPU-only data prep, and W&B logging", not yet like a cleanly separated `public`-artifact recipe.

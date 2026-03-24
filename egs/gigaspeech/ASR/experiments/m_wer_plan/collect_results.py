#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
BATCH_RE = re.compile(r"Epoch (\d+), batch (\d+).*batch size: (\d+)")
MEM_RE = re.compile(r"Maximum memory allocated so far is (\d+)MB")
AVG_RE = re.compile(r"avg-(\d+)")
BEAM_RE = re.compile(r"beam_size_(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to egs/gigaspeech/ASR",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for generated CSV/Markdown summaries",
    )
    return parser.parse_args()


def safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def parse_wer_filename(path: Path) -> Tuple[str, str, str]:
    # Filename format: wer-summary-{test_set}-{key}-{suffix}.txt
    # suffix is usually epoch-... or iter-...
    stem = path.stem
    prefix = "wer-summary-"
    if not stem.startswith(prefix):
        return "unknown", "unknown", "unknown"
    rest = stem[len(prefix) :]
    if "-" not in rest:
        return rest, "unknown", "unknown"
    test_set, rem = rest.split("-", 1)

    pos = rem.rfind("-epoch-")
    if pos < 0:
        pos = rem.rfind("-iter-")
    if pos < 0:
        return test_set, rem, "unknown"

    key = rem[:pos]
    suffix = rem[pos + 1 :]
    return test_set, key, suffix


def iter_wer_rows(asr_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in asr_dir.rglob("wer-summary-*.txt"):
        if "zipformer" not in p.as_posix():
            continue

        test_set, key, suffix = parse_wer_filename(p)
        decode_method = p.parent.name
        exp_dir = p.parent.parent

        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("settings"):
                    continue
                if "\t" not in line:
                    continue
                setting, wer_str = line.split("\t", 1)
                try:
                    wer = float(wer_str)
                except ValueError:
                    continue
                rows.append(
                    {
                        "exp_dir": safe_rel(exp_dir, asr_dir),
                        "decode_method": decode_method,
                        "test_set": test_set,
                        "key": key,
                        "setting": setting,
                        "suffix": suffix,
                        "wer": f"{wer:.4f}",
                        "wer_float": wer,
                        "source_file": safe_rel(p, asr_dir),
                    }
                )
    return rows


def parse_run_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_time(line: str) -> Optional[dt.datetime]:
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_train_log(log_path: Path) -> Dict[str, Optional[float]]:
    first_ts: Optional[dt.datetime] = None
    last_ts: Optional[dt.datetime] = None
    prev_epoch = None
    prev_batch = None
    cumulative_batches = 0
    batch_sizes: List[int] = []
    max_mem_mb = 0
    batch_lines = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = parse_time(line)
            if t is not None:
                if first_ts is None:
                    first_ts = t
                last_ts = t

            m_batch = BATCH_RE.search(line)
            if m_batch:
                epoch = int(m_batch.group(1))
                batch = int(m_batch.group(2))
                bsz = int(m_batch.group(3))
                batch_lines += 1
                batch_sizes.append(bsz)
                if prev_epoch is not None and prev_batch is not None:
                    if epoch == prev_epoch and batch >= prev_batch:
                        cumulative_batches += batch - prev_batch
                    elif epoch > prev_epoch:
                        # Underestimates true count around epoch boundaries,
                        # but keeps a stable monotonic approximation.
                        cumulative_batches += batch
                prev_epoch = epoch
                prev_batch = batch

            m_mem = MEM_RE.search(line)
            if m_mem:
                max_mem_mb = max(max_mem_mb, int(m_mem.group(1)))

    elapsed_s = None
    if first_ts is not None and last_ts is not None:
        elapsed_s = (last_ts - first_ts).total_seconds()
        if elapsed_s < 0:
            elapsed_s = None

    avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else None
    batches_per_sec = (
        (cumulative_batches / elapsed_s)
        if (elapsed_s is not None and elapsed_s > 0 and cumulative_batches > 0)
        else None
    )
    samples_per_sec = (
        (batches_per_sec * avg_batch_size)
        if (batches_per_sec is not None and avg_batch_size is not None)
        else None
    )

    return {
        "elapsed_s": elapsed_s,
        "cumulative_batches_approx": float(cumulative_batches),
        "batches_per_sec_approx": batches_per_sec,
        "avg_batch_size_logged": avg_batch_size,
        "samples_per_sec_approx": samples_per_sec,
        "max_gpu_mem_mb": float(max_mem_mb) if max_mem_mb else None,
        "batch_log_lines": float(batch_lines),
    }


def aggregate_train_metrics(asr_dir: Path, exp_dirs: Iterable[str]) -> List[Dict[str, str]]:
    metrics_rows: List[Dict[str, str]] = []
    for exp_rel in sorted(set(exp_dirs)):
        exp_path = asr_dir / exp_rel
        log_dir = exp_path / "log"
        if not log_dir.is_dir():
            continue

        total_elapsed = 0.0
        total_batches = 0.0
        total_batch_lines = 0.0
        weighted_avg_batch_num = 0.0
        max_mem = 0.0

        log_files = sorted(log_dir.glob("log-train*"))
        if not log_files:
            continue

        for logf in log_files:
            m = parse_train_log(logf)
            elapsed = m["elapsed_s"] or 0.0
            batches = m["cumulative_batches_approx"] or 0.0
            avg_bsz = m["avg_batch_size_logged"] or 0.0
            batch_lines = m["batch_log_lines"] or 0.0
            mem = m["max_gpu_mem_mb"] or 0.0

            total_elapsed += elapsed
            total_batches += batches
            total_batch_lines += batch_lines
            weighted_avg_batch_num += avg_bsz * batch_lines
            max_mem = max(max_mem, mem)

        if total_batch_lines > 0:
            avg_batch_size = weighted_avg_batch_num / total_batch_lines
        else:
            avg_batch_size = None

        batches_per_sec = (
            total_batches / total_elapsed if total_elapsed > 0 and total_batches > 0 else None
        )
        samples_per_sec = (
            batches_per_sec * avg_batch_size
            if batches_per_sec is not None and avg_batch_size is not None
            else None
        )

        metrics_rows.append(
            {
                "exp_dir": exp_rel,
                "train_elapsed_s_approx": f"{total_elapsed:.2f}" if total_elapsed else "",
                "cumulative_batches_approx": f"{total_batches:.2f}" if total_batches else "",
                "batches_per_sec_approx": f"{batches_per_sec:.6f}" if batches_per_sec else "",
                "avg_batch_size_logged": f"{avg_batch_size:.4f}" if avg_batch_size else "",
                "samples_per_sec_approx": f"{samples_per_sec:.4f}" if samples_per_sec else "",
                "max_gpu_mem_mb": f"{max_mem:.0f}" if max_mem else "",
                "log_files_count": str(len(log_files)),
            }
        )

    return metrics_rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_suffix_avg(suffix: str) -> Optional[int]:
    m = AVG_RE.search(suffix)
    return int(m.group(1)) if m else None


def parse_setting_beam(setting: str) -> Optional[int]:
    m = BEAM_RE.search(setting)
    return int(m.group(1)) if m else None


def build_summary_md(
    out_path: Path,
    wer_rows: List[Dict[str, str]],
    run_rows: List[Dict[str, str]],
    train_rows: List[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# GigaSpeech M WER Summary")
    lines.append("")
    lines.append("## Top WER (Dev/Test)")
    lines.append("")
    lines.append("| rank | test_set | WER | decode_method | setting | suffix | exp_dir |")
    lines.append("|---:|---|---:|---|---|---|---|")

    top = sorted(wer_rows, key=lambda x: x["wer_float"])[:30]
    for i, r in enumerate(top, start=1):
        lines.append(
            f"| {i} | {r['test_set']} | {r['wer_float']:.4f} | {r['decode_method']} | {r['setting']} | {r['suffix']} | {r['exp_dir']} |"
        )

    test_rows = [r for r in wer_rows if r["test_set"] == "test"]
    research_best = min(test_rows, key=lambda x: x["wer_float"]) if test_rows else None

    deploy_pool = []
    for r in test_rows:
        if r["decode_method"] != "modified_beam_search":
            continue
        beam = parse_setting_beam(r["setting"])
        avg = parse_suffix_avg(r["suffix"])
        use_avg_model = "use-averaged-model" in r["suffix"]
        if beam is not None and avg is not None and beam <= 4 and avg <= 9 and use_avg_model:
            deploy_pool.append(r)
    deployment_best = min(deploy_pool, key=lambda x: x["wer_float"]) if deploy_pool else None

    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    if research_best is not None:
        lines.append(
            f"- Research candidate (best test WER): `{research_best['wer_float']:.4f}` | `{research_best['decode_method']}` | `{research_best['setting']}` | `{research_best['suffix']}` | `{research_best['exp_dir']}`"
        )
    else:
        lines.append("- Research candidate: N/A")

    if deployment_best is not None:
        lines.append(
            f"- Deployment candidate (beam<=4, avg<=9, averaged-model): `{deployment_best['wer_float']:.4f}` | `{deployment_best['setting']}` | `{deployment_best['suffix']}` | `{deployment_best['exp_dir']}`"
        )
    else:
        lines.append("- Deployment candidate: no row matched the constrained criteria")

    lines.append("")
    lines.append("## Run Event Files")
    lines.append("")
    lines.append(f"- Baseline/Ablation/Decode sweep event rows: `{len(run_rows)}`")
    lines.append(f"- Training metrics rows: `{len(train_rows)}`")
    lines.append(f"- WER rows: `{len(wer_rows)}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    asr_dir = args.asr_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = parse_run_csv(results_dir / "baseline_runs.csv")
    ablation_rows = parse_run_csv(results_dir / "ablation_runs.csv")
    decode_sweep_rows = parse_run_csv(results_dir / "decode_sweep_runs.csv")
    run_rows = baseline_rows + ablation_rows + decode_sweep_rows

    wer_rows = iter_wer_rows(asr_dir)
    write_csv(results_dir / "wer_rows.csv", [{k: v for k, v in r.items() if k != "wer_float"} for r in wer_rows])
    write_csv(results_dir / "run_events.csv", run_rows)

    exp_dirs = set()
    for r in run_rows:
        exp = r.get("exp_dir", "")
        if exp:
            exp_dirs.add(exp)
    for r in wer_rows:
        exp_dirs.add(r["exp_dir"])

    train_rows = aggregate_train_metrics(asr_dir, exp_dirs)
    write_csv(results_dir / "train_metrics.csv", train_rows)

    build_summary_md(
        results_dir / "summary.md",
        wer_rows=wer_rows,
        run_rows=run_rows,
        train_rows=train_rows,
    )

    print(f"Wrote: {results_dir / 'wer_rows.csv'}")
    print(f"Wrote: {results_dir / 'run_events.csv'}")
    print(f"Wrote: {results_dir / 'train_metrics.csv'}")
    print(f"Wrote: {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()


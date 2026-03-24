#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUB_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*->\s*(.+?)\s*$")
ONE_TOKEN_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
TOTAL_RE = re.compile(
    r"Errors:\s+(\d+)\s+insertions,\s+(\d+)\s+deletions,\s+(\d+)\s+substitutions,"
)


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
        "--pattern",
        type=str,
        default="zipformer/exp_m_*/**/errs-*.txt",
        help="Glob pattern under --asr-dir",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Top-K entries in markdown report",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "error_analysis.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "error_analysis.md",
        help="Markdown output path",
    )
    return parser.parse_args()


def normalize_token(tok: str) -> str:
    return tok.strip()


def is_digit_token(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)


def is_liaison_like(tok: str) -> bool:
    return ("'" in tok) or ("-" in tok)


def is_proper_noun_like(tok: str) -> bool:
    # Best-effort heuristic:
    # TitleCase words are treated as proper-noun-like.
    # This can miss entities in all-caps transcripts.
    if not tok:
        return False
    letters = "".join(ch for ch in tok if ch.isalpha())
    if not letters:
        return False
    return letters[:1].isupper() and letters[1:].islower()


def parse_err_file(path: Path):
    section = None
    subs = Counter()
    dels = Counter()
    ins = Counter()
    total_ins = total_del = total_sub = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m_total = TOTAL_RE.search(line)
            if m_total:
                total_ins += int(m_total.group(1))
                total_del += int(m_total.group(2))
                total_sub += int(m_total.group(3))

            if line.startswith("SUBSTITUTIONS:"):
                section = "sub"
                continue
            if line.startswith("DELETIONS:"):
                section = "del"
                continue
            if line.startswith("INSERTIONS:"):
                section = "ins"
                continue
            if line.startswith("PER-WORD STATS:"):
                section = None
                continue
            if not line.strip():
                continue

            if section == "sub":
                m = SUB_RE.match(line)
                if not m:
                    continue
                c = int(m.group(1))
                ref = normalize_token(m.group(2))
                hyp = normalize_token(m.group(3))
                subs[(ref, hyp)] += c
            elif section == "del":
                m = ONE_TOKEN_RE.match(line)
                if not m:
                    continue
                c = int(m.group(1))
                ref = normalize_token(m.group(2))
                dels[ref] += c
            elif section == "ins":
                m = ONE_TOKEN_RE.match(line)
                if not m:
                    continue
                c = int(m.group(1))
                hyp = normalize_token(m.group(2))
                ins[hyp] += c

    return {
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "total_ins": total_ins,
        "total_del": total_del,
        "total_sub": total_sub,
    }


def counter_top(counter: Counter, topk: int):
    return [{"item": k, "count": v} for k, v in counter.most_common(topk)]


def pair_counter_top(counter: Counter, topk: int):
    rows = []
    for (ref, hyp), c in counter.most_common(topk):
        rows.append({"ref": ref, "hyp": hyp, "count": c})
    return rows


def flatten_sub_tokens(subs: Counter) -> Iterable[Tuple[str, int]]:
    for (ref, hyp), c in subs.items():
        yield ref, c
        yield hyp, c


def category_counts(subs: Counter, dels: Counter, ins: Counter) -> Dict[str, int]:
    cat = defaultdict(int)

    for tok, c in flatten_sub_tokens(subs):
        if is_digit_token(tok):
            cat["digit"] += c
        if is_proper_noun_like(tok):
            cat["proper_noun_like"] += c
        if is_liaison_like(tok):
            cat["liaison_like"] += c

    for tok, c in dels.items():
        if is_digit_token(tok):
            cat["digit"] += c
        if is_proper_noun_like(tok):
            cat["proper_noun_like"] += c
        if is_liaison_like(tok):
            cat["liaison_like"] += c

    for tok, c in ins.items():
        if is_digit_token(tok):
            cat["digit"] += c
        if is_proper_noun_like(tok):
            cat["proper_noun_like"] += c
        if is_liaison_like(tok):
            cat["liaison_like"] += c

    return dict(cat)


def write_md(
    out_path: Path,
    num_files: int,
    total_ins: int,
    total_del: int,
    total_sub: int,
    cat: Dict[str, int],
    top_sub: List[Dict[str, str]],
    top_del: List[Dict[str, str]],
    top_ins: List[Dict[str, str]],
):
    total_err = total_ins + total_del + total_sub
    lines = []
    lines.append("# Error Analysis")
    lines.append("")
    lines.append(f"- Parsed error files: `{num_files}`")
    lines.append(f"- Insertions: `{total_ins}`")
    lines.append(f"- Deletions: `{total_del}`")
    lines.append(f"- Substitutions: `{total_sub}`")
    lines.append(f"- Total: `{total_err}`")
    lines.append("")
    lines.append("## Heuristic Categories")
    lines.append("")
    lines.append("| category | count | ratio_of_total_err |")
    lines.append("|---|---:|---:|")
    for k in sorted(cat.keys()):
        ratio = (cat[k] / total_err * 100.0) if total_err > 0 else 0.0
        lines.append(f"| {k} | {cat[k]} | {ratio:.2f}% |")
    if not cat:
        lines.append("| N/A | 0 | 0.00% |")
    lines.append("")
    lines.append("## Top Substitutions")
    lines.append("")
    lines.append("| count | ref | hyp |")
    lines.append("|---:|---|---|")
    for r in top_sub:
        lines.append(f"| {r['count']} | {r['ref']} | {r['hyp']} |")
    if not top_sub:
        lines.append("| 0 | N/A | N/A |")
    lines.append("")
    lines.append("## Top Deletions")
    lines.append("")
    lines.append("| count | ref |")
    lines.append("|---:|---|")
    for r in top_del:
        lines.append(f"| {r['count']} | {r['item']} |")
    if not top_del:
        lines.append("| 0 | N/A |")
    lines.append("")
    lines.append("## Top Insertions")
    lines.append("")
    lines.append("| count | hyp |")
    lines.append("|---:|---|")
    for r in top_ins:
        lines.append(f"| {r['count']} | {r['item']} |")
    if not top_ins:
        lines.append("| 0 | N/A |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `digit`: token contains at least one numeric char.")
    lines.append("- `proper_noun_like`: token matches TitleCase heuristic.")
    lines.append("- `liaison_like`: token contains apostrophe or hyphen.")
    lines.append("- These are best-effort text-only proxies, not linguistic tagging.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    asr_dir = args.asr_dir.resolve()
    paths = sorted(asr_dir.glob(args.pattern))
    # Support recursive glob under pattern containing **.
    if not paths:
        paths = sorted(asr_dir.rglob("errs-*.txt"))
        paths = [p for p in paths if p.match(str((asr_dir / args.pattern).resolve()).replace(str(asr_dir), "**"))]

    # If strict pattern matching above returns empty, fallback to a direct recursive
    # search in zipformer experiments.
    if not paths:
        paths = sorted(
            p for p in asr_dir.rglob("errs-*.txt") if "zipformer/exp_m_" in p.as_posix()
        )

    agg_subs = Counter()
    agg_dels = Counter()
    agg_ins = Counter()
    total_ins = total_del = total_sub = 0

    for p in paths:
        parsed = parse_err_file(p)
        agg_subs.update(parsed["subs"])
        agg_dels.update(parsed["dels"])
        agg_ins.update(parsed["ins"])
        total_ins += parsed["total_ins"]
        total_del += parsed["total_del"]
        total_sub += parsed["total_sub"]

    cat = category_counts(agg_subs, agg_dels, agg_ins)
    top_sub = pair_counter_top(agg_subs, args.topk)
    top_del = counter_top(agg_dels, args.topk)
    top_ins = counter_top(agg_ins, args.topk)

    payload = {
        "files_parsed": len(paths),
        "total_insertions": total_ins,
        "total_deletions": total_del,
        "total_substitutions": total_sub,
        "total_errors": total_ins + total_del + total_sub,
        "category_counts_heuristic": cat,
        "top_substitutions": top_sub,
        "top_deletions": top_del,
        "top_insertions": top_ins,
        "source_files": [str(p) for p in paths],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(
        args.out_md,
        num_files=len(paths),
        total_ins=total_ins,
        total_del=total_del,
        total_sub=total_sub,
        cat=cat,
        top_sub=top_sub,
        top_del=top_del,
        top_ins=top_ins,
    )

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    print(f"Parsed files: {len(paths)}")


if __name__ == "__main__":
    main()


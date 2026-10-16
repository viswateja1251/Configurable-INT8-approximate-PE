#!/usr/bin/env python3
"""
compare_runs.py

Compare two OpenLane metrics snapshots — e.g. an exact-arithmetic PE run
vs. an approximate-arithmetic PE run — and report the differences.

Usage:
    python3 compare_runs.py <baseline> <compare> \
        [--baseline-label exact] [--compare-label approx] [--csv out.csv] [--all]

Each of <baseline>/<compare> may be either:
  - a path directly to a metrics.json or or_metrics_out.json file,
  - an OpenLane run directory (…/runs/<tag>), or
  - a design directory (…/openlane/pe_exact) containing several run tags.

For a run directory or design directory, the script prefers a finished
run's final/metrics.json. If a run has no final/ (e.g. it ended in a
deferred signoff failure, such as unmet setup timing, before reaching
the last step), it falls back to the or_metrics_out.json of that run's
highest-numbered completed step — the cumulative metrics snapshot every
step writes — so you can still compare a design that hasn't closed
timing yet. When a design directory has several run tags, any run with
a final/metrics.json is preferred (most recently modified); only if
none do does it fall back to partial step metrics (also most recently
modified).

The script always prints which file it used for each side, and flags
clearly whenever a side is a PARTIAL (non-final) snapshot, including
which step it came from — the later the step, the more complete the
numbers (a snapshot from just before the last step is essentially
final; one from early synthesis is pre-layout only and not really
comparable to a finished run).

By default, only metrics that look like area, cell count, timing slack,
power, or sign-off violations are shown — the numbers people usually
care about when comparing two implementations. Pass --all to see every
numeric metric found in either file, including ones only present in one.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HIGHLIGHT_SUBSTRINGS = (
    "area", "count", "util", "slack", "ws", "wns", "tns", "power",
    "violation", "drc", "lvs", "antenna", "slew",
)

STEP_DIR_RE = re.compile(r"^(\d+)-")


class MetricsNotFound(Exception):
    def __init__(self, path, checked):
        self.path = path
        self.checked = checked
        msg = (
            f"Could not find any metrics for '{path}'.\n"
            "Checked:\n  " + "\n  ".join(checked) + "\n"
            "This usually means the run hasn't produced any step output "
            "yet (e.g. it failed during setup, before step 01)."
        )
        super().__init__(msg)


def looks_like_run_dir(d: Path) -> bool:
    if (d / "final").is_dir():
        return True
    try:
        return any(STEP_DIR_RE.match(c.name) for c in d.iterdir() if c.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        return False


def iter_run_dirs(path: Path):
    """Yield OpenLane run directories to consider for `path`.

    If `path` itself looks like a run directory, yield just it.
    Otherwise treat it as a design directory and yield each run under
    its runs/ subfolder.
    """
    if path.is_dir() and looks_like_run_dir(path):
        yield path
        return
    runs_dir = path / "runs"
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                yield d


def highest_step_metrics(run_dir: Path):
    """Return the or_metrics_out.json of the highest-numbered step under
    run_dir that actually has one, or None if there isn't one."""
    step_dirs = [d for d in run_dir.iterdir() if d.is_dir() and STEP_DIR_RE.match(d.name)]
    step_dirs.sort(key=lambda d: int(STEP_DIR_RE.match(d.name).group(1)), reverse=True)
    for d in step_dirs:
        candidate = d / "or_metrics_out.json"
        if candidate.is_file():
            return candidate
    return None


def find_best_metrics(path: Path):
    """Resolve `path` to a metrics file.

    Returns (metrics_path, is_final, step_label_or_None).
    Raises MetricsNotFound if nothing usable is found.
    """
    checked = []

    if path.is_file():
        is_final = path.name == "metrics.json"
        return path, is_final, None

    run_dirs = list(iter_run_dirs(path))
    if not run_dirs:
        checked.append(f"{path}  (not a file, not a run dir, no runs/ subfolder)")
        raise MetricsNotFound(path, checked)

    # Prefer a finished run: final/metrics.json, most recently modified.
    final_candidates = []
    for d in run_dirs:
        f = d / "final" / "metrics.json"
        checked.append(str(f))
        if f.is_file():
            final_candidates.append(f)
    if final_candidates:
        best = max(final_candidates, key=lambda p: p.stat().st_mtime)
        return best, True, None

    # Fall back to the highest-numbered completed step in each run dir.
    partial_candidates = []
    for d in run_dirs:
        m = highest_step_metrics(d)
        if m:
            checked.append(f"{m}  (partial)")
            partial_candidates.append(m)
        else:
            checked.append(f"{d}/<NN-step>/or_metrics_out.json  (no step metrics found)")
    if partial_candidates:
        best = max(partial_candidates, key=lambda p: p.stat().st_mtime)
        step_label = best.parent.name  # e.g. "74-misc-reportmanufacturability"
        return best, False, step_label

    raise MetricsNotFound(path, checked)


def load_metrics(label: str, raw_path: str):
    path = Path(raw_path)
    try:
        metrics_path, is_final, step_label = find_best_metrics(path)
    except MetricsNotFound as e:
        sys.exit(f"[{label}] {e}")

    if is_final:
        print(f"[{label}] using: {metrics_path}  (final)")
    else:
        print(
            f"[{label}] using: {metrics_path}\n"
            f"  -> PARTIAL: this run never reached final/ (stopped after "
            f"step '{step_label}'). Numbers are a snapshot, not signed-off results."
        )

    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(
            f"[{label}] {metrics_path} is not valid JSON ({e}).\n"
            "This can happen if the run was killed while writing this "
            "file (e.g. an out-of-memory kill) — rerun the flow."
        )
    return data, is_final


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def build_rows(base: dict, comp: dict, show_all: bool):
    rows = []
    for key in sorted(set(base) | set(comp)):
        bv, cv = base.get(key), comp.get(key)
        if not ((is_number(bv) or bv is None) and (is_number(cv) or cv is None)):
            continue  # skip non-numeric metrics (strings, lists, nested objects)
        if bv is None and cv is None:
            continue
        if not show_all and not any(s in key.lower() for s in HIGHLIGHT_SUBSTRINGS):
            continue
        delta = (cv - bv) if (is_number(bv) and is_number(cv)) else None
        pct = (100 * delta / bv) if (delta is not None and bv not in (0, None)) else None
        rows.append((key, bv, cv, delta, pct))
    return rows


def print_table(rows, baseline_label, compare_label):
    label_w = max(len(k) for k, *_ in rows)
    header = (
        f"{'metric':<{label_w}}  {baseline_label:>14}  {compare_label:>14}  "
        f"{'delta':>12}  {'% change':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for key, bv, cv, delta, pct in rows:
        bv_s = f"{bv:.4g}" if is_number(bv) else "\u2014"
        cv_s = f"{cv:.4g}" if is_number(cv) else "\u2014"
        delta_s = f"{delta:+.4g}" if delta is not None else "\u2014"
        pct_s = f"{pct:+.1f}%" if pct is not None else "\u2014"
        print(f"{key:<{label_w}}  {bv_s:>14}  {cv_s:>14}  {delta_s:>12}  {pct_s:>10}")


def write_csv(rows, path, baseline_label, compare_label):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", baseline_label, compare_label, "delta", "pct_change"])
        for key, bv, cv, delta, pct in rows:
            w.writerow([key, bv, cv, delta, pct])
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="Baseline run (e.g. exact) — run dir, design dir, or metrics file")
    ap.add_argument("compare", help="Comparison run (e.g. approx) — run dir, design dir, or metrics file")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--compare-label", default="compare")
    ap.add_argument("--csv", metavar="FILE", help="Write the full comparison table to a CSV file")
    ap.add_argument("--all", action="store_true", help="Show every numeric metric, not just the highlighted set")
    args = ap.parse_args()

    base, base_final = load_metrics(args.baseline_label, args.baseline)
    comp, comp_final = load_metrics(args.compare_label, args.compare)

    rows = build_rows(base, comp, args.all)
    if not rows:
        sys.exit("No comparable numeric metrics found (try --all, or check both files parsed correctly).")

    if not (base_final and comp_final):
        print(
            "\nNote: at least one side is a PARTIAL snapshot (run didn't finish). "
            "Treat timing/signoff metrics with extra caution below."
        )

    print_table(rows, args.baseline_label, args.compare_label)

    if args.csv:
        write_csv(rows, args.csv, args.baseline_label, args.compare_label)


if __name__ == "__main__":
    main()

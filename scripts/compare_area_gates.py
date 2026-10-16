#!/usr/bin/env python3
"""
compare_area_gates.py

Quick, focused comparison of total area and gate (cell) count between
two OpenLane runs — e.g. an exact-arithmetic PE vs. an approximate one.

For a full metric comparison (timing, power, violations, etc.), use
compare_runs.py instead. This script only prints area and gate-count
rows, so the result is a glance-able handful of lines.

Usage:
    python3 compare_area_gates.py <baseline> <compare> \
        [--baseline-label exact] [--compare-label approx] [--list-keys]

Each of <baseline>/<compare> may be a run directory, a design directory
(with several run tags), or a metrics file directly — same resolution
as compare_runs.py: a finished run's final/metrics.json is preferred;
if a run never reached final/ (e.g. it ended in a deferred signoff
failure such as unmet setup timing), the highest-numbered completed
step's or_metrics_out.json is used instead, and the output says so.

Area and gate-count metrics are found by matching key names rather
than a hardcoded list, since the exact metric names can vary slightly
between OpenLane versions. If nothing matches, pass --list-keys to see
every numeric key actually present in both files and tell me which
ones are the real area/count metrics so the match can be tightened.
"""
import argparse
import json
import re
import sys
from pathlib import Path

STEP_DIR_RE = re.compile(r"^(\d+)-")

GATE_COUNT_EXCLUDE = (
    "error", "violation", "disconnected", "unmapped", "critical", "overlap", "pin",
)


class MetricsNotFound(Exception):
    def __init__(self, path, checked):
        self.path = path
        self.checked = checked
        msg = (
            f"Could not find any metrics for '{path}'.\n"
            "Checked:\n  " + "\n  ".join(checked) + "\n"
            "This usually means the run hasn't produced any step output "
            "yet (e.g. it failed before step 01)."
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
    if path.is_dir() and looks_like_run_dir(path):
        yield path
        return
    runs_dir = path / "runs"
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                yield d


def highest_step_metrics(run_dir: Path):
    """Return the metrics file of the highest-numbered step under run_dir
    that has one, or None if there isn't one. Different OpenLane builds
    write different filenames per step -- or_metrics_out.json on some,
    state_out.json (with metrics nested under a "metrics" key) on others
    -- so both are checked."""
    step_dirs = [d for d in run_dir.iterdir() if d.is_dir() and STEP_DIR_RE.match(d.name)]
    step_dirs.sort(key=lambda d: int(STEP_DIR_RE.match(d.name).group(1)), reverse=True)
    for d in step_dirs:
        for name in ("or_metrics_out.json", "state_out.json"):
            candidate = d / name
            if candidate.is_file():
                return candidate
    return None


def find_best_metrics(path: Path):
    checked = []

    if path.is_file():
        is_final = path.name == "metrics.json"
        return path, is_final, None

    run_dirs = list(iter_run_dirs(path))
    if not run_dirs:
        checked.append(f"{path}  (not a file, not a run dir, no runs/ subfolder)")
        raise MetricsNotFound(path, checked)

    final_candidates = []
    for d in run_dirs:
        f = d / "final" / "metrics.json"
        checked.append(str(f))
        if f.is_file():
            final_candidates.append(f)
    if final_candidates:
        best = max(final_candidates, key=lambda p: p.stat().st_mtime)
        return best, True, None

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
        return best, False, best.parent.name

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
            f"  -> PARTIAL: run never reached final/ (stopped after step '{step_label}')."
        )

    try:
        data = json.loads(metrics_path.read_text())
        # state_out.json nests the real metrics under a "metrics" key;
        # metrics.json / or_metrics_out.json are already flat, so this
        # is a no-op for those.
        if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
            data = data["metrics"]
    except json.JSONDecodeError as e:
        sys.exit(f"[{label}] {metrics_path} is not valid JSON ({e}).")
    return data


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_area_key(key: str) -> bool:
    return "area" in key.lower()


def is_gate_count_key(key: str) -> bool:
    k = key.lower()
    if "count" not in k:
        return False
    if not any(s in k for s in ("instance", "cell")):
        return False
    if any(s in k for s in GATE_COUNT_EXCLUDE):
        return False
    return True


def build_rows(base: dict, comp: dict, predicate):
    rows = []
    for key in sorted(set(base) | set(comp)):
        if not predicate(key):
            continue
        bv, cv = base.get(key), comp.get(key)
        if not ((is_number(bv) or bv is None) and (is_number(cv) or cv is None)):
            continue
        if bv is None and cv is None:
            continue
        delta = (cv - bv) if (is_number(bv) and is_number(cv)) else None
        pct = (100 * delta / bv) if (delta is not None and bv not in (0, None)) else None
        rows.append((key, bv, cv, delta, pct))
    return rows


def print_section(title, rows, baseline_label, compare_label):
    print(f"\n{title}")
    if not rows:
        print("  (none found — try --list-keys to see what's actually in these files)")
        return
    label_w = max(len(k) for k, *_ in rows)
    header = (
        f"  {'metric':<{label_w}}  {baseline_label:>14}  {compare_label:>14}  "
        f"{'delta':>12}  {'% change':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, bv, cv, delta, pct in rows:
        bv_s = f"{bv:.4g}" if is_number(bv) else "\u2014"
        cv_s = f"{cv:.4g}" if is_number(cv) else "\u2014"
        delta_s = f"{delta:+.4g}" if delta is not None else "\u2014"
        pct_s = f"{pct:+.1f}%" if pct is not None else "\u2014"
        print(f"  {key:<{label_w}}  {bv_s:>14}  {cv_s:>14}  {delta_s:>12}  {pct_s:>10}")


def write_csv(path, area_rows, gate_rows, baseline_label, compare_label):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "metric", baseline_label, compare_label, "delta", "pct_change"])
        for key, bv, cv, delta, pct in area_rows:
            w.writerow(["area", key, bv, cv, delta, pct])
        for key, bv, cv, delta, pct in gate_rows:
            w.writerow(["gate_count", key, bv, cv, delta, pct])
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="Baseline run (e.g. exact) — run dir, design dir, or metrics file")
    ap.add_argument("compare", help="Comparison run (e.g. approx) — run dir, design dir, or metrics file")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--compare-label", default="compare")
    ap.add_argument("--csv", metavar="FILE", help="Write the area and gate-count rows to a CSV file")
    ap.add_argument(
        "--list-keys", action="store_true",
        help="Also print every numeric key found in either file (useful if area/gate rows come up empty)",
    )
    args = ap.parse_args()

    base = load_metrics(args.baseline_label, args.baseline)
    comp = load_metrics(args.compare_label, args.compare)

    area_rows = build_rows(base, comp, is_area_key)
    gate_rows = build_rows(base, comp, is_gate_count_key)

    print_section("Area", area_rows, args.baseline_label, args.compare_label)
    print_section("Gate / cell count", gate_rows, args.baseline_label, args.compare_label)

    if args.csv:
        write_csv(args.csv, area_rows, gate_rows, args.baseline_label, args.compare_label)

    if args.list_keys:
        print("\nAll numeric keys found:")
        all_keys = sorted(
            k for k in (set(base) | set(comp))
            if is_number(base.get(k)) or is_number(comp.get(k))
        )
        for k in all_keys:
            print(f"  {k}")


if __name__ == "__main__":
    main()
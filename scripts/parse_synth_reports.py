#!/usr/bin/env python3
"""
parse_synth_report.py

Parse the Yosys `stat` report embedded in an OpenLane synthesis log
(the block that starts with "=== <module> ===") and extract:

  - total chip area (um^2)
  - total cell count
  - a per-gate-type breakdown
  - a rough breakdown by logic family (sequential / combinational / etc.)

Works on:
  - a synthesis log file directly, e.g.
      openlane/pe/runs/RUN_.../*-yosys-synthesis/*.log
  - an OpenLane run directory (the script will search it for the
    synthesis log automatically)

Usage:
    python3 parse_synth_report.py <log_file_or_run_dir> [options]

Options:
    --all         Show every stat block found in the log (Yosys often
                  prints one after generic synthesis and another after
                  technology mapping). Default: only the LAST block,
                  which is the final, technology-mapped result.
    --json FILE   Write the parsed data to a JSON file.
    --csv FILE    Write the gate-level breakdown to a CSV file.

Note: this parses the human-readable log text, so exact wording can
shift slightly between Yosys versions. If a run's log doesn't parse,
paste a few lines around "=== " and the "Chip area" line and the
regexes below can be adjusted.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# --- Regexes for the Yosys `stat` report block ------------------------

# Splits the log into one chunk per "=== module_name ===" header.
STAT_BLOCK_RE = re.compile(
    r"===\s*(?P<module>\S+)\s*===\s*\n(?P<body>.*?)(?=\n===|\Z)",
    re.DOTALL,
)

# Lines like "   Number of cells:                    3"
NUM_LINE_RE = re.compile(r"^\s*Number of ([\w ]+?):\s+(\d+)\s*$")

# Lines like "     sky130_fd_sc_hd__inv_2               1"
# (exactly two whitespace-separated tokens: a cell name, then a count)
CELL_LINE_RE = re.compile(r"^\s+(\S+)\s+(\d+)\s*$")

# Lines like "   Chip area for module '\inv_test': 23.947200"
AREA_RE = re.compile(
    r"Chip area for (?:top module|module) '\\?([\w.]+)':\s+([\d.]+)"
)


def find_synthesis_log(run_dir: Path) -> Path:
    """Locate the Yosys synthesis step log inside an OpenLane run directory."""
    candidates = sorted(run_dir.glob("*yosys-synthesis*/*.log"))
    if not candidates:
        candidates = sorted(run_dir.rglob("*synthesis*.log"))
    if not candidates:
        raise FileNotFoundError(
            f"No synthesis log found under {run_dir}. "
            "Pass the log file itself instead of the run directory."
        )
    # If several match, the synthesis step usually sorts first; take the
    # first one, since later steps (e.g. resizer) also emit their own logs.
    return candidates[0]


def parse_stat_report(text: str):
    """Parse every `stat` block in the log text.

    Returns a list of dicts: {module, summary, cells, area}
    """
    blocks = []
    for match in STAT_BLOCK_RE.finditer(text):
        module = match.group("module")
        body = match.group("body")
        summary, cells = {}, {}
        for line in body.splitlines():
            num_match = NUM_LINE_RE.match(line)
            if num_match:
                summary[num_match.group(1).strip()] = int(num_match.group(2))
                continue
            cell_match = CELL_LINE_RE.match(line)
            if cell_match:
                cells[cell_match.group(1)] = int(cell_match.group(2))
        blocks.append({"module": module, "summary": summary, "cells": cells, "area": None})

    # "Chip area" lines sit just after their block, not inside it, so
    # match them separately and attach to the nearest block with the
    # same module name.
    for area_match in AREA_RE.finditer(text):
        module, area = area_match.group(1), float(area_match.group(2))
        for block in reversed(blocks):
            if block["module"].lstrip("\\") == module.lstrip("\\"):
                block["area"] = area
                break

    return [b for b in blocks if b["cells"]]  # drop empty/non-stat blocks


def classify_cell(cell_name: str) -> str:
    """Rough classification of a standard-cell name into a logic family.
    Based on common Sky130 / Yosys generic-cell naming conventions —
    adjust if your PDK uses different suffixes."""
    name = cell_name.lower()
    if any(tag in name for tag in ("dfx", "dfr", "dfs", "sdfx", "sdfr", "edfx", "latch", "$_dff", "$_sdff")):
        return "sequential"
    if "mux" in name or "$_mux" in name:
        return "mux"
    if "buf" in name:
        return "buffer"
    if "inv" in name or "$_not_" in name:
        return "inverter"
    if any(tag in name for tag in ("fill", "tap", "decap", "diode")):
        return "physical-only"
    if any(tag in name for tag in ("and", "or", "nand", "nor", "xor", "xnor", "aoi", "oai")):
        return "combinational logic"
    return "other"


def summarize(block: dict) -> dict:
    cells = block["cells"]
    total_cells = sum(cells.values())
    by_family = {}
    for cell, count in cells.items():
        family = classify_cell(cell)
        by_family[family] = by_family.get(family, 0) + count

    return {
        "module": block["module"],
        "area_um2": block["area"],
        "total_cells": total_cells,
        "cell_breakdown": dict(sorted(cells.items(), key=lambda kv: -kv[1])),
        "family_breakdown": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "wires": block["summary"].get("wires"),
        "wire_bits": block["summary"].get("wire bits"),
    }


def print_report(summary: dict) -> None:
    print(f"Module: {summary['module']}")
    if summary["area_um2"] is not None:
        print(f"Chip area: {summary['area_um2']:.4f} um^2")
    else:
        print("Chip area: not found in log")
    print(f"Total cells: {summary['total_cells']}")
    if summary["wires"] is not None:
        print(f"Wires: {summary['wires']} ({summary.get('wire_bits')} bits)")
    print()
    print("Family breakdown:")
    for family, count in summary["family_breakdown"].items():
        pct = 100 * count / summary["total_cells"] if summary["total_cells"] else 0
        print(f"  {family:<20} {count:>6}  ({pct:5.1f}%)")
    print()
    print("Gate-level breakdown:")
    for cell, count in summary["cell_breakdown"].items():
        print(f"  {cell:<40} {count:>6}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Synthesis log file, or an OpenLane run directory")
    ap.add_argument("--all", action="store_true", help="Show every stat block found, not just the last one")
    ap.add_argument("--json", metavar="FILE", help="Write parsed data to a JSON file")
    ap.add_argument("--csv", metavar="FILE", help="Write the gate breakdown to a CSV file")
    args = ap.parse_args()

    path = Path(args.path)
    log_path = find_synthesis_log(path) if path.is_dir() else path
    if path.is_dir():
        print(f"Using log: {log_path}\n")

    text = log_path.read_text(errors="ignore")
    blocks = parse_stat_report(text)
    if not blocks:
        sys.exit("No Yosys `stat` report found in this log.")

    selected = blocks if args.all else [blocks[-1]]
    summaries = [summarize(b) for b in selected]

    for i, summary in enumerate(summaries):
        if len(summaries) > 1:
            print(f"--- Stat block {i + 1} of {len(summaries)} ({'final/mapped' if i == len(summaries) - 1 else 'earlier pass'}) ---")
        print_report(summary)
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=2))
        print(f"Wrote {args.json}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["module", "cell_type", "count"])
            for summary in summaries:
                for cell, count in summary["cell_breakdown"].items():
                    writer.writerow([summary["module"], cell, count])
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
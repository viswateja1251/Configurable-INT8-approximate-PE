import re
import sys
from pathlib import Path

def parse_yosys_stat(rpt_path):
    metrics = {
        "num_cells": 0,
        "area": 0.0,
        "cell_types": {}
    }
    
    if not Path(rpt_path).is_file():
        print(f"Error: {rpt_path} not found.")
        return None

    with open(rpt_path, "r") as f:
        for line in f:
            # Match cell count
            m_cells = re.search(r"Number of cells:\s+(\d+)", line)
            if m_cells:
                metrics["num_cells"] = int(m_cells.group(1))

            # Match chip area in um^2
            m_area = re.search(r"Chip area for module .*: ([\d\.]+)", line)
            if m_area:
                metrics["area"] = float(m_area.group(1))

            # Match individual standard cell breakdown
            m_gate = re.search(r"^\s+([a-zA-Z0-9_]+)\s+(\d+)", line)
            if m_gate and not line.strip().startswith("Number"):
                cell_name, count = m_gate.group(1), int(m_gate.group(2))
                metrics["cell_types"][cell_name] = count

    return metrics

def main():
    exact = parse_yosys_stat("reports/synth_pe_exact.rpt")
    approx = parse_yosys_stat("reports/synth_pe_approx.rpt")

    if not exact or not approx:
        sys.exit(1)

    area_reduction = ((exact["area"] - approx["area"]) / exact["area"]) * 100 if exact["area"] > 0 else 0.0
    cell_reduction = ((exact["num_cells"] - approx["num_cells"]) / exact["num_cells"]) * 100 if exact["num_cells"] > 0 else 0.0

    print("=" * 60)
    print(f"{'Metric':<25}{'pe_exact':<15}{'pe_approx':<15}{'Reduction':<10}")
    print("=" * 60)
    print(f"{'Total Cell Count':<25}{exact['num_cells']:<15}{approx['num_cells']:<15}{cell_reduction:.2f}%")
    print(f"{'Gate-Level Area (um²)':<25}{exact['area']:<15.2f}{approx['area']:<15.2f}{area_reduction:.2f}%")
    print("-" * 60)

    # Verification against target reduction range
    if 25.0 <= area_reduction <= 40.0:
        print(f"[PASS] Area reduction is {area_reduction:.2f}%, within the target 25-40% range.")
    else:
        print(f"[NOTE] Area reduction is {area_reduction:.2f}% (Target: 25-40%).")

if __name__ == "__main__":
    main()
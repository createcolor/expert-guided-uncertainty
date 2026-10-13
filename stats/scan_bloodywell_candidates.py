"""Rank candidate BloodyWell uncertainty files by how well they reproduce a saved curve.

Used to find which saved file produced a paper curve when several versions exist.

Reads candidate paths from stdin (e.g. from grep -rl). Run from the repo root:
    grep -rl --include="*.json" "05. CcEeCwKk_3_1.png" /alpha 2>/dev/null | \
        python -m stats.scan_bloodywell_candidates <test_markup> <misclass.json> <saved_curve.json>
"""
import json
import sys
from collections import Counter

import numpy as np

from stats.collect_bloodywell import wrong_matrix
from stats.sample_curves import make_grid, net_curves, area_above_curve


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_uncertainty_file(data, wells):
    """[N] uncertainties if ``data`` is {number: [wells]} with every test well exactly once, else a reason."""
    if not isinstance(data, dict) or not data:
        return "not a dict"
    per_well = {}
    for key, names in data.items():
        try:
            unc = float(key)
        except ValueError:
            return "non-numeric keys"
        if not isinstance(names, list):
            return "values are not lists"
        for name in names:
            if name in per_well:
                return "a well appears more than once"
            per_well[name] = unc
    if any(w not in per_well for w in wells):
        return "does not cover the test wells"
    return np.array([per_well[w] for w in wells])


def main():
    markup_path, misclass_path, curve_path = sys.argv[1:4]
    wells = list(load(markup_path))
    wrong = wrong_matrix(load(misclass_path), wells, 10)
    saved = np.array(list(load(curve_path).values()))
    grid = make_grid(len(saved) - 1)
    saved_aac = float(np.trapz(1 - saved[:801, 0], grid[:801]))

    results, skipped = [], Counter()
    for path in (line.strip() for line in sys.stdin):
        if not path:
            continue
        try:
            u = as_uncertainty_file(load(path), wells)
        except (OSError, ValueError, UnicodeDecodeError):
            u = "unreadable JSON"
        if isinstance(u, str):
            skipped[u] += 1
            continue
        curves = net_curves(u, wrong, grid)
        diff = max(np.abs(curves.mean(0) - saved[:, 0]).max(), np.abs(curves.std(0) - saved[:, 1]).max())
        results.append((diff, float(area_above_curve(curves, grid).mean()), path))

    print(f"Saved curve AAC: {saved_aac * 1e3:.4f} x1e-3")
    print(f"Skipped (not an uncertainty file): {dict(skipped)}")
    print(f"{len(results)} candidate(s), best first:")
    for diff, aac, path in sorted(results)[:20]:
        mark = "  <-- MATCH" if diff < 1e-9 else ""
        print(f"  max|diff| {diff:.1e}  AAC {aac * 1e3:.4f}  {path}{mark}")


if __name__ == "__main__":
    main()

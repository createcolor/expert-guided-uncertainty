"""Paired tests of one method against chosen comparators on every dataset.

For each dataset, all methods are scored on the same bootstrap resamples of
test items (questions, wells or images), and each comparison reports
``Delta AAC = AAC(ours) - AAC(other)`` (negative = ours better) with a
percentile CI and a two-sided bootstrap p-value. For segmentation, where AAC is
a mean over images, a Wilcoxon signed-rank test on per-image Delta AAC is
reported as well. p-values are Holm-corrected over all comparisons in the run.

Before testing, each dataset's full-test-set AAC is checked against its
``expected_aac`` (the AACs of the original runs, see ``stats.expected_aac``); a
mismatch raises ``ReproductionError`` and nothing is reported. It is then
compared with ``paper_aac`` (the paper's tables, rounded); differences are
printed as warnings.

Usage:
    python -m stats.run_paired_tests -c stats/configs/paired_tests.json -o results/paired_tests
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from stats.data import ItemLevelData, ImageLevelData
from stats.paired_stats import (
    PairedResult, bootstrap_method_stats, paired_difference, holm, wilcoxon_p,
)
from stats.sample_curves import make_grid, method_aac


def _available(dataset: dict, labels: list[str], stored: set[str]) -> dict[str, str]:
    """``{display label: stored method name}`` for the labels this dataset provides."""
    found = {}
    for label in labels:
        name = dataset["methods"].get(label)
        if name is None:
            print(f"  [{dataset['name']}] no method mapped for '{label}', skipping it.")
        elif name not in stored:
            print(f"  [{dataset['name']}] WARNING: '{name}' ({label}) is not in {dataset['data']}, "
                  "skipping it.")
        else:
            found[label] = name
    return found


class ReproductionError(RuntimeError):
    """The collected data does not reproduce the AAC of the paper's original run."""


def check_reproduction(dataset: dict, methods: dict[str, str], observed: dict[str, float],
                       rtol: float, atol: float) -> None:
    """Compare full-test-set AAC with the dataset's ``expected_aac`` (keyed by stored method name).

    Raises ``ReproductionError`` on any mismatch so that no statistics are reported
    for data that differs from what the paper's numbers were computed on.
    """
    expected = dataset.get("expected_aac")
    if not expected:
        print(f"  [{dataset['name']}] no expected_aac given -- reproduction NOT checked.")
        return
    failures = []
    for label, name in methods.items():
        if name not in expected:
            print(f"  [{dataset['name']}] {label}: no expected value, not checked.")
            continue
        got, want = observed[label], expected[name]
        ok = abs(got - want) <= atol + rtol * abs(want)
        print(f"  [{dataset['name']}] {label}: AAC {got:.8f}, expected {want:.8f} "
              f"-> {'reproduced' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{label} ({name}): got {got:.8f}, expected {want:.8f}")
    if failures:
        raise ReproductionError(f"[{dataset['name']}] collected data does not reproduce the "
                                "saved run:\n  " + "\n  ".join(failures))


def compare_with_paper(dataset: dict, methods: dict[str, str], observed: dict[str, float],
                       atol: float) -> list[str]:
    """Compare full-test-set AAC with the values printed in the paper (``paper_aac``).

    The paper rounds to two decimals (x1e-3), so this is a looser, informative
    check: differences above ``atol`` are reported, not raised. Returns them.
    """
    printed = dataset.get("paper_aac") or {}
    off = []
    for label, name in methods.items():
        if name in printed and abs(observed[label] - printed[name]) > atol:
            off.append(f"{label}: {observed[label] * 1e3:.4f} vs paper {printed[name] * 1e3:.2f} (x1e-3)")
    for line in off:
        print(f"  [{dataset['name']}] WARNING, differs from the paper's table: {line}")
    if printed and not off:
        print(f"  [{dataset['name']}] all AACs agree with the paper's table (within {atol:.0e}).")
    return off


def item_level(dataset: dict, labels: list[str], config: dict):
    """Per-item data -> (stat_fn, n_units, per-unit AAC or None, {label: stored name})."""
    data = ItemLevelData.load(dataset["data"])
    methods = _available(dataset, labels, set(data.uncertainty))
    grid = make_grid(config.get("n_bins", 1000))
    r_max = config.get("r_max", 0.8)

    def stat(idx: np.ndarray) -> dict[str, float]:
        wrong = data.wrong[:, idx]
        return {label: method_aac(data.uncertainty[name][..., idx], wrong, grid, r_max)
                for label, name in methods.items()}
    return stat, data.wrong.shape[1], None, methods


def image_level(dataset: dict, labels: list[str], config: dict):
    """Per-image data -> (stat_fn, n_units, per-image AAC per method, {label: stored name}).

    Per-image AAC is ``area(oracle) - area(method)`` averaged over nets (the
    paper's convention); without an oracle it is ``r_max - area(method)``.
    Either way differences between methods are unaffected.
    """
    data = ImageLevelData.load(dataset["data"])
    methods = _available(dataset, labels, set(data.image_auc))
    oracle = dataset.get("oracle")
    reference = (data.image_auc[oracle].mean(axis=0) if oracle
                 else np.full(next(iter(data.image_auc.values())).shape[1], data.r_max))
    per_image = {label: reference - data.image_auc[name].mean(axis=0)
                 for label, name in methods.items()}

    def stat(idx: np.ndarray) -> dict[str, float]:
        return {label: float(values[idx].mean()) for label, values in per_image.items()}
    return stat, len(reference), per_image, methods


def run(config: dict) -> list[PairedResult]:
    ours, comparators = config["ours"], config["comparators"]
    rtol, atol = config.get("reproduction_rtol", 1e-4), config.get("reproduction_atol", 1e-7)
    results = []
    for dataset in config["datasets"]:
        builder = item_level if dataset["level"] == "item" else image_level
        stat, n_units, per_unit, methods = builder(dataset, [ours, *comparators], config)

        print(f"{dataset['name']}: checking reproduction of the saved run...")
        full = stat(np.arange(n_units))
        check_reproduction(dataset, methods, full, rtol, atol)
        compare_with_paper(dataset, methods, full, config.get("paper_atol", 1e-5))

        print(f"{dataset['name']}: bootstrapping ({config.get('n_boot', 10_000)} resamples)...")
        observed, boot = bootstrap_method_stats(stat, n_units, config.get("n_boot", 10_000),
                                                config.get("seed", 0))
        if ours not in observed:
            raise KeyError(f"[{dataset['name']}] '{ours}' (ours) is missing.")
        for other in comparators:
            if other not in observed:
                continue
            delta, low, high, p_boot = paired_difference(observed, boot, ours, other,
                                                         config.get("ci", 0.95))
            p_wil = wilcoxon_p(per_unit[ours] - per_unit[other]) if per_unit else None
            results.append(PairedResult(dataset["name"], ours, other, n_units,
                                        observed[ours], observed[other],
                                        delta, low, high, p_boot, p_wil))

    for result, adjusted in zip(results, holm([r.p_bootstrap for r in results])):
        result.p_holm = adjusted
    return results


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def markdown_table(results: list[PairedResult], scale: float, alpha: float) -> str:
    unit = f"×10^{-int(round(np.log10(scale)))}" if scale != 1 else ""
    lines = [
        f"| Dataset | Ours vs. | n | AAC ours | AAC other | ΔAAC {unit} [95% CI] "
        f"| p (bootstrap) | p (Wilcoxon) | p (Holm) |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        mark = " *" if r.p_holm < alpha else ""
        lines.append(
            f"| {r.dataset} | {r.other} | {r.n_units} | {r.aac_ours * scale:.2f} | "
            f"{r.aac_other * scale:.2f} | {r.delta * scale:+.2f} "
            f"[{r.ci_low * scale:+.2f}, {r.ci_high * scale:+.2f}] | {_fmt_p(r.p_bootstrap)} | "
            f"{_fmt_p(r.p_wilcoxon)} | {_fmt_p(r.p_holm)}{mark} |")
    lines.append("")
    lines.append(f"ΔAAC = AAC(ours) − AAC(other); negative means ours is better. "
                 f"* = significant at α = {alpha} after Holm correction "
                 f"over all {len(results)} comparisons.")
    return "\n".join(lines)


def save(results: list[PairedResult], out_dir: Path, table: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.as_dict() for r in results]
    with open(out_dir / "paired_tests.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "paired_tests.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "paired_tests.md").write_text(table + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", type=Path, required=True, help="Paired-tests config JSON.")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Directory for the results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    results = run(config)
    table = markdown_table(results, config.get("scale", 1e3), config.get("alpha", 0.05))
    if args.out is not None:
        save(results, args.out, table)
    try:                                   # non-UTF-8 consoles cannot print "Δ"
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    print()
    print(table)
    if args.out is not None:
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

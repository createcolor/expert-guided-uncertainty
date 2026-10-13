"""Collect per-well uncertainties and mistakes for BloodyWell.

The paper's BloodyWell curves were not produced by the QA pipeline but by the
classification ``rejection_curves.py`` (``tasks/classification/rejection_curves.py``
here, ``uncertain_classification/rejection_curves.py`` in the old repo). It reads

  - one uncertainty file per method, ``{uncertainty: [well names]}`` (or one
    such file per base net), saved as ``unc_<method>.json``;
  - the base nets' mistakes, ``{net_id: [misclassified well names]}``, written
    by the classification ``inference.py`` as ``<nets>_misclass.json``;
  - the test markup, ``{well name: {..., "gt_result": 0/1}}``.

Those files already hold everything per well, so nothing is recomputed. Wells
with equal uncertainty are discarded together. The saved paper curves were
made with ties taken at full precision (``TV_CE``, ``TV_gt_exps`` and ``CE``
reproduce exactly without rounding), which is the default here. Some saved
curves merged near-equal values (``experts_p(1-p)`` needs ``round(u, 3..10)``;
the classification ``rejection_curves.py`` itself rounds to 7 digits); use
``--round-digits METHOD=N`` for those.

Usage:
    python -m stats.collect_bloodywell \
        --test-markup tasks/classification/markup/BloodyWell/test_dataset_10_10_80.json \
        --misclass /path/to/MobileNet_small_10_10_80_meta_8_misclass.json \
        --uncertainty-dir /path/to/uncertainty-nlp/uncertainty_estimation/results/bloodywell \
        --methods "TV_pred_exps_p(1-p)_sum_CE_var" "TV_CE_p(1-p)_sum_CE_var" \
                  "TV_gt_exps_p(1-p)_sum_CE_var" \
        --validate-dir /path/to/uncertainty-nlp-2/uncertainty_estimation/results/bloodywell/rejection_curves \
        --out results/paired_tests/bloodywell_items.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stats.data import ItemLevelData
from stats.sample_curves import make_grid, net_curves

def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def misclass_from_outputs(outputs: dict, test_markup: dict, threshold: float = 0.5) -> dict:
    """``{net_id: [wells]}`` as ``inference.save_statistics`` writes it (``thr_path=None`` -> 0.5)."""
    misclass = {}
    for net_id, net_outputs in outputs.items():
        misclass[net_id] = [well for well, output in net_outputs.items()
                            if (1 if float(output) >= threshold else 0) != test_markup[well]["gt_result"]]
    return misclass


def wrong_matrix(misclass: dict, wells: list[str], n_nets: int) -> np.ndarray:
    """``[nets, N]``: whether net ``k`` (key ``str(k)``, as the old code reads it) got each well wrong."""
    missing = [str(net) for net in range(n_nets) if str(net) not in misclass]
    if missing:
        raise KeyError(f"Mistakes file has no entries for nets {missing}; it has {sorted(misclass)}.")
    return np.array([np.isin(wells, list(misclass[str(net)])) for net in range(n_nets)])


def ranking_from_file(unc_to_wells: dict, wells: list[str], source: Path,
                      digits: int | None = None) -> np.ndarray:
    """``[N]`` uncertainty per test well, optionally rounded to ``digits`` (which sets the ties)."""
    per_well = {}
    for unc, names in unc_to_wells.items():
        for name in names:
            if name in per_well:
                raise ValueError(f"{source.name}: well {name} appears more than once.")
            per_well[name] = float(unc) if digits is None else round(float(unc), digits)
    missing = [w for w in wells if w not in per_well]
    if missing:
        # The old code never discards such wells; that case is not reproduced here.
        raise ValueError(f"{source.name}: {len(missing)} test wells have no uncertainty, "
                         f"e.g. {missing[:3]}.")
    return np.array([per_well[w] for w in wells])


def method_files(uncertainty_dir: Path, method: str, n_nets: int) -> list[Path]:
    """``unc_<method>.json`` (one ranking) or ``unc_<method>/*_<net>.json`` (one ranking per net)."""
    single = uncertainty_dir / f"unc_{method}.json"
    if single.is_file():
        return [single]
    per_net_dir = uncertainty_dir / f"unc_{method}"
    if per_net_dir.is_dir():
        files = [next(iter(sorted(per_net_dir.glob(f"*_{net}.json"))), None) for net in range(n_nets)]
        if all(files):
            return files
        raise FileNotFoundError(
            f"{per_net_dir} should hold one file per base net (*_0.json ... *_{n_nets - 1}.json); "
            f"it has {sorted(p.name for p in per_net_dir.glob('*.json'))[:6]} ...")
    raise FileNotFoundError(f"Neither {single} nor {per_net_dir} exists.")


def validate(method: str, uncertainty: np.ndarray, wrong: np.ndarray, rc_path: Path,
             tol: float = 1e-9) -> None:
    """Compare with the saved ``{rate: [mean, std]}`` curve of the original run."""
    saved = _load(rc_path)
    grid = make_grid(len(saved) - 1)
    curves = net_curves(uncertainty, wrong, grid)
    saved_mean = np.array([v[0] for v in saved.values()])
    saved_std = np.array([v[1] for v in saved.values()])
    diff = max(np.abs(curves.mean(0) - saved_mean).max(), np.abs(curves.std(0) - saved_std).max())
    if diff > tol:
        raise AssertionError(f"{method}: rejection curve differs from {rc_path} "
                             f"(max |diff| = {diff:.2e}).")
    print(f"  {method}: matches {rc_path.name} (max diff {diff:.1e})")


def collect(test_markup: dict, misclass: dict, uncertainty_dir: Path, methods: list[str],
            n_nets: int = 10, validate_dir: Path | None = None,
            round_digits: dict[str, int] | None = None,
            skip_validate: set[str] | None = None) -> ItemLevelData:
    wells = list(test_markup)
    round_digits = round_digits or {}
    skip_validate = skip_validate or set()
    data = ItemLevelData(item_ids=np.array(wells), wrong=wrong_matrix(misclass, wells, n_nets))
    for method in methods:
        files = method_files(uncertainty_dir, method, n_nets)
        rankings = [ranking_from_file(_load(path), wells, path, round_digits.get(method))
                    for path in files]
        data.uncertainty[method] = rankings[0] if len(rankings) == 1 else np.stack(rankings)
        if validate_dir is not None and method in skip_validate:
            print(f"  {method}: not validated against a saved curve (--skip-validate)")
        elif validate_dir is not None:
            validate(method, data.uncertainty[method], data.wrong, validate_dir / f"{method}.json")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-markup", type=Path, required=True, help="BloodyWell test markup JSON.")
    mistakes = parser.add_mutually_exclusive_group(required=True)
    mistakes.add_argument("--misclass", type=Path, help="<nets>_misclass.json: {net_id: [wells]}.")
    mistakes.add_argument("--outputs", type=Path,
                          help="<nets>_<ep>ep_outputs.json: {net_id: {well: output}}; mistakes are "
                               "rebuilt as inference.save_statistics does (threshold 0.5).")
    parser.add_argument("--uncertainty-dir", type=Path, required=True,
                        help="Directory with the saved unc_<method>.json files.")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--n-nets", type=int, default=10)
    parser.add_argument("--round-digits", nargs="*", default=[], metavar="METHOD=N",
                        help="Round a method's uncertainties to N digits before ranking (sets ties); "
                             "default: full precision.")
    parser.add_argument("--validate-dir", type=Path, default=None,
                        help="Saved rejection_curves dir of the original run (<method>.json).")
    parser.add_argument("--skip-validate", nargs="*", default=[], metavar="METHOD",
                        help="Methods whose saved curve is not from the paper's run (BloodyWell CAE); "
                             "run_paired_tests still checks their AAC against expected_aac.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_markup = _load(args.test_markup)
    misclass = (_load(args.misclass) if args.misclass is not None
                else misclass_from_outputs(_load(args.outputs), test_markup))
    round_digits = {m: int(n) for m, n in (item.rsplit("=", 1) for item in args.round_digits)}
    data = collect(test_markup, misclass, args.uncertainty_dir, args.methods, args.n_nets,
                   args.validate_dir, round_digits, set(args.skip_validate))
    data.save(args.out)
    print(f"Saved {len(data.item_ids)} wells x {data.wrong.shape[0]} nets, "
          f"{len(data.uncertainty)} method(s) to {args.out}")


if __name__ == "__main__":
    main()

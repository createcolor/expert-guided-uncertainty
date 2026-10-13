"""AAC values of the paper's original runs, computed from their saved artifacts.

``run_paired_tests`` checks that the collected per-item / per-image data
reproduces these values on the full test set before it reports any test, so a
mismatch in data, checkpoints or preprocessing fails loudly instead of silently
changing the statistics.

Sources:
  - item-level (PubMedQA, BloodyWell): ``rejection_curves_data/<method>.json`` written by
    the old ``plot_rejection_curves.get_rc_data`` (``{rate: [mean_acc, std]}``);
    AAC = integral of ``1 - accuracy`` over ``[0, r_max]``.
  - image-level (LIDC-IDRI, RIGA): ``rcs/<run>/<method>/dice_scores.npy`` (per-net curves);
    AAC = mean over nets of ``area(oracle) - area(method)`` over ``[0, r_max]``,
    as in the old ``aac_std.py``.

Usage:
    python -m stats.expected_aac item  --dir .../results/bert_v2/rejection_curves_data \
        --methods "TV_pred_exps_p(1-p)_sum_CE_var" mcmc
    python -m stats.expected_aac image --dir .../segmentation/rcs/e3_10k_microavg_std_full_dice \
        --methods tv_cae tv_base --oracle oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stats.sample_curves import area_above_curve
from stats.segmentation_curves import old_throwaway_rates, area_under_curves


def item_level_aac(rc_dir: Path, method: str, r_max: float = 0.8) -> float:
    with open(rc_dir / f"{method}.json", "r", encoding="utf-8") as f:
        curve = json.load(f)
    grid = np.array([float(rate) for rate in curve])
    values = np.array([float(v[0]) if isinstance(v, list) else float(v) for v in curve.values()])
    order = np.argsort(grid)
    return float(area_above_curve(values[order], grid[order], r_max)[0])


def image_level_aac(rcs_dir: Path, method: str, oracle: str | None, rates: np.ndarray,
                    r_max: float = 0.8) -> float:
    def areas(name: str) -> np.ndarray:
        curves = np.load(rcs_dir / name / "dice_scores.npy")
        return area_under_curves(curves.reshape(-1, curves.shape[-1]), rates, r_max)
    reference = areas(oracle) if oracle else r_max
    return float(np.mean(reference - areas(method)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("level", choices=["item", "image"])
    parser.add_argument("--dir", type=Path, required=True,
                        help="rejection_curves_data dir (item) or rcs/<run> dir (image).")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--oracle", default=None, help="Oracle method name (image level).")
    parser.add_argument("--r-max", type=float, default=0.8)
    parser.add_argument("--rates", type=float, nargs=3, default=[0.0001, 1.0, 10000],
                        metavar=("START", "STOP", "NUM"), help="Old rate grid (image level).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.level == "item":
        values = {m: item_level_aac(args.dir, m, args.r_max) for m in args.methods}
    else:
        rates = old_throwaway_rates(args.rates[0], args.rates[1], int(args.rates[2]))
        values = {m: image_level_aac(args.dir, m, args.oracle, rates, args.r_max)
                  for m in args.methods}
    print(json.dumps(values, indent=4))


if __name__ == "__main__":
    main()

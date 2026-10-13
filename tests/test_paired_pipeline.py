"""End-to-end checks for the paired-test pipeline on synthetic data.

The server run is trusted because two independent paths must agree:
  - ``stats.expected_aac`` computes AAC from the artifacts the original runs saved
    (net-averaged rejection-curve JSONs; per-net ``dice_scores.npy``);
  - ``stats.run_paired_tests`` computes AAC from the collected per-item / per-image
    data that the paired bootstrap resamples.
These tests show the two paths give the same AAC when the underlying data is the
same, and that ``run_paired_tests`` refuses to report when it is not.
"""

import json

import numpy as np
import pytest

from stats.data import ItemLevelData, ImageLevelData
from stats.expected_aac import item_level_aac, image_level_aac
from stats.run_paired_tests import run, ReproductionError
from stats.sample_curves import make_grid, net_curves, method_aac
from stats.segmentation_curves import old_throwaway_rates
from tests.test_paired_stats import _old_get_rc_info_averaging_mode, _to_old_format

torch = pytest.importorskip("torch")
from stats.collect_segmentation import image_areas  # noqa: E402  (needs torch)


# ── Synthetic data in the formats the old pipelines saved ─────────────────────

def _item_data(seed=0, n_items=120, n_nets=4):
    rng = np.random.default_rng(seed)
    wrong = rng.random((n_nets, n_items)) < 0.25
    signal = wrong.mean(axis=0)                       # informative uncertainty
    unc = {
        "ours": np.round(signal + rng.normal(0, 0.15, n_items), 3),
        "other": np.round(signal + rng.normal(0, 0.60, n_items), 3),
        "per_net": np.round(wrong + rng.normal(0, 0.5, wrong.shape), 3),
    }
    return ItemLevelData(item_ids=np.arange(n_items).astype(str), wrong=wrong, uncertainty=unc)


def _image_data(seed=0, n_nets=3, n_images=12, n_pixels=400):
    """Per-net per-image areas and mean curves, as collect_segmentation produces them."""
    rng = np.random.default_rng(seed)
    rates = old_throwaway_rates(0.0001, 1.0, 2000)
    labels = (rng.random((n_images, n_pixels)) < 0.3).astype(np.int64)
    data = ImageLevelData(rates=rates, r_max=0.8)
    # Same base-ensemble predictions for every method; only the uncertainty differs.
    preds = np.where(rng.random((n_nets, n_images, n_pixels)) < 0.15, 1 - labels, labels)
    wrong = (preds != labels).astype(float)
    for name, noise in [("ours", 0.2), ("other", 0.6), ("oracle", 0.0)]:
        unc = (wrong + rng.normal(0, noise, wrong.shape)).astype(np.float32)
        data.image_auc[name], data.mean_curve[name] = image_areas(preds, labels, unc, rates, 0.8)
    return data


# ── Expected (saved artifacts) vs observed (collected data) ───────────────────

class TestTwoPathsAgree:
    def test_item_level(self, tmp_path):
        data = _item_data()
        for name in ("ours", "per_net"):
            unc = data.uncertainty[name]
            old_unc = ([_to_old_format(unc[k], data.wrong, 3) for k in range(data.wrong.shape[0])]
                       if unc.ndim == 2 else _to_old_format(unc, data.wrong, 3))
            curve = _old_get_rc_info_averaging_mode(old_unc, 1000, range(data.wrong.shape[0]), 3)
            # the old get_rc_data file format: {rate: [mean, std]}
            saved = {str(round(i / 1000, 6)): [v, 0.0] for i, v in enumerate(curve)}
            (tmp_path / f"{name}.json").write_text(json.dumps(saved))
            assert item_level_aac(tmp_path, name) == pytest.approx(
                method_aac(unc, data.wrong, make_grid(1000)), abs=1e-12)

    def test_image_level(self, tmp_path):
        data = _image_data()
        for name, curves in data.mean_curve.items():
            (tmp_path / name).mkdir()
            np.save(tmp_path / name / "dice_scores.npy", curves[None])   # old shape [1, nets, R]
        expected = image_level_aac(tmp_path, "ours", "oracle", data.rates)
        observed = np.mean(data.image_auc["oracle"].mean(0) - data.image_auc["ours"].mean(0))
        assert observed == pytest.approx(expected, abs=1e-12)


# ── run_paired_tests ──────────────────────────────────────────────────────────

def _config(tmp_path, expected_item=None, expected_image=None):
    item, image = _item_data(), _image_data()
    item.save(tmp_path / "items.npz")
    image.save(tmp_path / "images.npz")
    return {
        "ours": "Ours", "comparators": ["Other", "Per-net"], "n_boot": 300, "seed": 0,
        "datasets": [
            {"name": "items", "level": "item", "data": str(tmp_path / "items.npz"),
             "methods": {"Ours": "ours", "Other": "other", "Per-net": "per_net"},
             "expected_aac": expected_item},
            {"name": "images", "level": "image", "data": str(tmp_path / "images.npz"),
             "oracle": "oracle", "methods": {"Ours": "ours", "Other": "other"},
             "expected_aac": expected_image},
        ],
    }, item, image


class TestRunPairedTests:
    def test_reports_when_reproduction_holds(self, tmp_path):
        _, item, image = _config(tmp_path)
        grid = make_grid(1000)
        expected_item = {n: method_aac(u, item.wrong, grid) for n, u in item.uncertainty.items()}
        expected_image = {n: float(np.mean(image.image_auc["oracle"].mean(0) - a.mean(0)))
                          for n, a in image.image_auc.items()}
        config, *_ = _config(tmp_path, expected_item, expected_image)
        results = run(config)

        assert [(r.dataset, r.other) for r in results] == [
            ("items", "Other"), ("items", "Per-net"), ("images", "Other")]
        assert all(r.p_holm is not None and r.p_holm >= r.p_bootstrap for r in results)
        assert results[0].p_wilcoxon is None and results[2].p_wilcoxon is not None
        # "ours" has the less noisy uncertainty in both datasets
        assert results[0].delta < 0 and results[2].delta < 0

    def test_refuses_to_report_on_mismatch(self, tmp_path):
        config, item, _ = _config(tmp_path)
        wrong_value = {"ours": method_aac(item.uncertainty["ours"], item.wrong, make_grid(1000)) + 1e-3}
        config["datasets"][0]["expected_aac"] = wrong_value
        with pytest.raises(ReproductionError):
            run(config)

    def test_missing_comparator_is_skipped(self, tmp_path):
        config, *_ = _config(tmp_path)
        del config["datasets"][0]["methods"]["Per-net"]
        results = run(config)
        assert ("items", "Per-net") not in [(r.dataset, r.other) for r in results]

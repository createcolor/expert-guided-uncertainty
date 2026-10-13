"""``stats.collect_bloodywell`` reproduces the classification rejection curves.

The ``_old_*`` functions are verbatim copies (docstrings removed) of
``tasks/classification/rejection_curves.py``, which produced the paper's
BloodyWell curves.
"""

import json
from typing import Dict, List

import numpy as np
import pytest

from stats.collect_bloodywell import collect, misclass_from_outputs


def _old_calculate_accuracy(misclas_alvs_names: List[str], test_alvs_names: List[str]):
    assert len(test_alvs_names) != 0
    mistakes_num = len(list(set(misclas_alvs_names) & set(test_alvs_names)))
    return 1.0 - mistakes_num / len(test_alvs_names)


def _old_get_accs_throwaway_rates(uncert_alv_name_dict: Dict, input_test_markup: Dict,
                                  misclas_alvs_names: List, reagent: str = "all"):
    test_markup = input_test_markup.copy()

    if reagent != "all":
        test_markup = {alv_name: alv_info for alv_name, alv_info in test_markup.items()
                       if alv_info["reagent"] == reagent}

        reagent_uncert_alv_name_dict = {}
        for unc, alv_names_list in uncert_alv_name_dict.items():
            for alv_name in alv_names_list:
                if alv_name in test_markup.keys():

                    if unc not in reagent_uncert_alv_name_dict:
                        reagent_uncert_alv_name_dict[unc] = []

                    reagent_uncert_alv_name_dict[unc].append(alv_name)

        uncert_alv_name_dict = reagent_uncert_alv_name_dict

    assert len(test_markup) > 0
    initial_acc = _old_calculate_accuracy(misclas_alvs_names, list(test_markup.keys()))
    accuracies, throwaway_rates = [initial_acc], [0.0]
    test_markup_left = test_markup.copy()
    init_alvs_number = len(test_markup)

    uncert_alv_name_dict_rounded = {}
    for rate in uncert_alv_name_dict:
        rounded_rate = round(float(rate), 7)
        if rounded_rate not in uncert_alv_name_dict_rounded:
            uncert_alv_name_dict_rounded[rounded_rate] = []
        uncert_alv_name_dict_rounded[rounded_rate] += uncert_alv_name_dict[rate]

    sorted_uncertainty_rates = sorted(list(map(float, uncert_alv_name_dict_rounded.keys())),
                                      reverse=True)

    throwaway_rate = 0.0

    for uncertainty_rate in sorted_uncertainty_rates:
        alvs2throwaway_names_candidates = uncert_alv_name_dict_rounded[uncertainty_rate]
        alvs2throwaway_names_list = [alv_name for alv_name in alvs2throwaway_names_candidates
                                     if alv_name in test_markup_left]
        if len(alvs2throwaway_names_list) != 0:
            for alv_name in alvs2throwaway_names_list:
                del test_markup_left[alv_name]

            if len(test_markup_left) > 0:
                throwaway_rate += len(alvs2throwaway_names_list) / init_alvs_number
                throwaway_rates.append(throwaway_rate)

                accuracy = _old_calculate_accuracy(misclas_alvs_names, test_markup_left)
                accuracies.append(accuracy)

                if accuracy == 1.0:
                    break

    accuracies.append(1.0)
    throwaway_rates.append(1.0)

    return accuracies, throwaway_rates


def _old_get_rc_info_averaging_mode(unc_alv_names_dicts: List[Dict], test_markup: Dict,
                                    n_bins: int = 20, reagent: str = "all",
                                    misclassified_alvs=None, ensemble_classifier_mode=False,
                                    path2save_markup=None, net_ids=range(10)):
    x_list = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ensemble_accuracies_dict = {x: [] for x in x_list}

    for net_id in net_ids:
        if len(unc_alv_names_dicts) > 1:
            unc_alv_names_dict = unc_alv_names_dicts[net_id]
        else:
            unc_alv_names_dict = unc_alv_names_dicts[0]

        if not ensemble_classifier_mode:
            misclas_alvs = misclassified_alvs[str(net_id)]
        else:
            misclas_alvs = misclassified_alvs

        accuracies, throwaway_rates = _old_get_accs_throwaway_rates(
            unc_alv_names_dict, test_markup, misclas_alvs, reagent=reagent)

        for x in x_list:
            if x in throwaway_rates:
                ensemble_accuracies_dict[x].append(accuracies[throwaway_rates.index(x)])
            else:
                throwaway_rates_less_x = list(filter(lambda rate: rate < x, throwaway_rates))
                left_closest_rate = min(throwaway_rates_less_x, key=lambda rate: x - rate)
                right_closest_rate = throwaway_rates[throwaway_rates.index(left_closest_rate) + 1]
                alpha = (x - left_closest_rate) / (right_closest_rate - left_closest_rate)

                accuracy1 = accuracies[throwaway_rates.index(left_closest_rate)]
                accuracy2 = accuracies[throwaway_rates.index(right_closest_rate)]
                accuracy = accuracy1 + alpha * (accuracy2 - accuracy1)

                ensemble_accuracies_dict[x].append(accuracy)

    ensemble_accuracies = {x: (np.mean(ensemble_accuracies_dict[x]),
                           np.std(ensemble_accuracies_dict[x])) for x in x_list}

    if path2save_markup is not None:
        with open(path2save_markup, "w", encoding="utf-8") as ensemble_file:
            json.dump(ensemble_accuracies, ensemble_file, indent=4)

    return ensemble_accuracies


# ── Synthetic BloodyWell run in the old file formats ─────────────────────────

N_NETS = 4


def _unc_file(rng, wells, wrong_any):
    """{str(u): [wells]}, with near-ties that only merge after round(u, 7)."""
    base = rng.choice(np.linspace(0.001, 0.05, 25), size=len(wells)) + 0.02 * wrong_any
    jitter = rng.integers(0, 3, size=len(wells)) * 1e-10
    out = {}
    for well, u in zip(wells, base + jitter):
        out.setdefault(str(float(u)), []).append(well)
    return out


@pytest.fixture
def old_run(tmp_path):
    rng = np.random.default_rng(0)
    wells = [f"{i:02d}. Card_{i % 7}_{i % 5}.png" for i in range(150)]
    markup = {w: {"gt_result": int(rng.integers(2)), "reagent": "c"} for w in wells}
    outputs = {str(net): {w: str(float(rng.random())) for w in wells} for net in range(N_NETS)}
    misclass = misclass_from_outputs(outputs, markup)
    wrong_any = np.array([any(w in misclass[str(n)] for n in range(N_NETS)) for w in wells])

    unc_dir, rc_dir = tmp_path / "unc", tmp_path / "rcs"
    (unc_dir / "unc_mcmc").mkdir(parents=True)
    rc_dir.mkdir()
    shared = _unc_file(rng, wells, wrong_any)
    (unc_dir / "unc_tv.json").write_text(json.dumps(shared))
    per_net = [_unc_file(rng, wells, wrong_any) for _ in range(N_NETS)]
    for net, unc in enumerate(per_net):
        (unc_dir / "unc_mcmc" / f"mcmc_{net}.json").write_text(json.dumps(unc))

    for name, dicts in [("tv", [shared]), ("mcmc", per_net)]:
        _old_get_rc_info_averaging_mode(dicts, markup, n_bins=1000, misclassified_alvs=misclass,
                                        path2save_markup=rc_dir / f"{name}.json",
                                        net_ids=range(N_NETS))
    return markup, misclass, unc_dir, rc_dir


def test_reproduces_saved_curves(old_run):
    markup, misclass, unc_dir, rc_dir = old_run
    data = collect(markup, misclass, unc_dir, ["tv", "mcmc"], N_NETS, validate_dir=rc_dir)
    assert data.uncertainty["tv"].shape == (150,)
    assert data.uncertainty["mcmc"].shape == (N_NETS, 150)
    assert data.wrong.shape == (N_NETS, 150)


def test_validation_catches_other_mistakes(old_run):
    markup, misclass, unc_dir, rc_dir = old_run
    other = {net: wells[1:] for net, wells in misclass.items()}
    with pytest.raises(AssertionError, match="differs"):
        collect(markup, other, unc_dir, ["tv"], N_NETS, validate_dir=rc_dir)


def test_missing_well_is_an_error(old_run):
    markup, misclass, unc_dir, _ = old_run
    markup = {**markup, "extra.png": {"gt_result": 0}}
    with pytest.raises(ValueError, match="no uncertainty"):
        collect(markup, misclass, unc_dir, ["tv"], N_NETS)


def test_misclass_from_outputs_uses_threshold_half():
    markup = {"a": {"gt_result": 1}, "b": {"gt_result": 0}, "c": {"gt_result": 1}}
    outputs = {"0": {"a": "0.5", "b": "0.5", "c": "0.49"}}
    assert misclass_from_outputs(outputs, markup) == {"0": ["b", "c"]}

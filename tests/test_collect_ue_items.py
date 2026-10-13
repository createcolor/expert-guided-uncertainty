"""``stats.collect_ue_items`` with model outputs saved as JSON (BloodyWell) or pickle (PubMedQA).

The old pipeline only reads outputs with ``load_pickle``, indexes nets by int
(``outputs[0]``) and finds MCMC samples with ``glob("*.pkl")``. The stand-in
module below does the same, so JSON outputs only work if the collector converts
them to the pickles the old code expects.
"""

import json
import pickle
import sys
import textwrap

import numpy as np
import pytest

from stats.collect_ue_items import collect, pickle_json_outputs

OLD_MODULE = textwrap.dedent('''
    import json, pickle
    from pathlib import Path
    import numpy as np

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_pickle(path):
        with open(str(path), "rb") as f:
            return pickle.load(f)

    def calc_ensemble_uncs(per_nets, mistakes, path, figs):
        out = {}
        for item in mistakes[0]:
            probas = np.array([np.array(net[item]["probas"]) for net in per_nets.values()])
            unc = str(round(float(np.sum(np.var(probas, axis=0))), figs))
            out.setdefault(unc, []).append([mistakes[net][item] for net in mistakes])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)

    def calculate_uncertainties(config, mistakes):
        figs, out = config["sign_figs_num"], Path(config["path2save_uncertainties"])
        out.mkdir(parents=True, exist_ok=True)
        if "CE_var" in config["UE_methods"]:
            outputs = load_pickle(config["path2outputs_epist"])
            outputs[0]                                    # nets are indexed by int
            calc_ensemble_uncs(outputs, mistakes, out / "CE_var.json", figs)
        if "mcmc" in config["UE_methods"]:
            ens = {}
            for path in Path(config["UE_methods"]["mcmc"]["path2outputs"]).glob("*.pkl"):
                for net, outputs in load_pickle(path).items():
                    ens.setdefault(net, {})[path.stem] = outputs
            (out / "mcmc").mkdir(exist_ok=True)
            for net, samples in ens.items():
                calc_ensemble_uncs(samples, mistakes, out / "mcmc" / f"mcmc_{net}.json", figs)
''')


def _outputs(rng, items, n_nets=3, n_classes=3):
    outputs = {}
    for net in range(n_nets):
        probas = rng.dirichlet(np.ones(n_classes), size=len(items))
        outputs[net] = {item: {"gt": int(rng.integers(n_classes)), "pred": int(p.argmax()),
                               "probas": p.tolist()} for item, p in zip(items, probas)}
    return outputs


def _write(outputs, path, fmt):
    if fmt == "json":
        path.write_text(json.dumps(outputs))           # net ids become "0", "1", ...
    else:
        path.write_bytes(pickle.dumps(outputs))


def _old_repo(tmp_path, fmt, seed=0):
    rng = np.random.default_rng(seed)
    repo = tmp_path / fmt
    (repo / "mcmc").mkdir(parents=True)
    (repo / "fake_old_ue.py").write_text(OLD_MODULE)
    items = [f"well_{i}" for i in range(40)]
    (repo / "test_markup.json").write_text(json.dumps({item: {} for item in items}))
    _write(_outputs(rng, items), repo / f"ens.{fmt}", fmt)
    for sample in range(4):
        _write(_outputs(rng, items), repo / "mcmc" / f"sample_{sample}.{fmt}", fmt)
    config = {
        "path2outputs_classifier": f"ens.{fmt}", "path2outputs_epist": f"ens.{fmt}",
        "test_markup": "test_markup.json", "sign_figs_num": 6,
        "path2save_uncertainties": "unused",
        "UE_methods": {"CE_var": {"single_net_file_name_template": None},
                       "mcmc": {"path2outputs": "mcmc", "single_net_file_name_template": "mcmc"}},
    }
    (repo / "config_UE.json").write_text(json.dumps(config))
    return repo


@pytest.fixture
def isolated(monkeypatch):
    """collect() chdirs into the old repo and prepends it to sys.path."""
    monkeypatch.chdir(".")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "fake_old_ue", raising=False)


def test_json_outputs_give_the_same_result_as_pickles(tmp_path, isolated):
    results = {}
    for fmt in ("pkl", "json"):
        repo = _old_repo(tmp_path, fmt)
        sys.modules.pop("fake_old_ue", None)
        results[fmt] = collect(repo, "fake_old_ue", repo / "config_UE.json", None, None)

    pkl, js = results["pkl"], results["json"]
    np.testing.assert_array_equal(pkl.item_ids, js.item_ids)
    np.testing.assert_array_equal(pkl.wrong, js.wrong)
    assert set(js.uncertainty) == {"CE_var", "mcmc"}
    assert js.uncertainty["mcmc"].shape == (3, 40)                 # one map per base net
    for method in pkl.uncertainty:
        np.testing.assert_array_equal(pkl.uncertainty[method], js.uncertainty[method])


def test_only_json_model_outputs_are_converted(tmp_path):
    repo = _old_repo(tmp_path, "json")
    config = json.loads((repo / "config_UE.json").read_text())
    config = {k: (str(repo / v) if isinstance(v, str) and k != "sign_figs_num" else v)
              for k, v in config.items()}
    config["UE_methods"]["mcmc"]["path2outputs"] = str(repo / "mcmc")
    converted = pickle_json_outputs(config, tmp_path / "converted")

    assert converted["test_markup"] == config["test_markup"]       # markup stays JSON
    assert converted["path2outputs_epist"] == converted["path2outputs_classifier"]  # converted once
    with open(converted["path2outputs_epist"], "rb") as f:
        assert list(pickle.load(f)) == [0, 1, 2]                    # int net ids, as in the pickles
    mcmc_dir = converted["UE_methods"]["mcmc"]["path2outputs"]
    assert sorted(p.name for p in (tmp_path / "converted").glob("*/*.pkl")
                  if str(p.parent) == mcmc_dir) == [f"sample_{i}.pkl" for i in range(4)]

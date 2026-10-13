"""Collect per-item uncertainties from the old ``uncertainty_estimation`` pipeline.

Used for PubMedQA and BloodyWell, whose paper results came from
``uncertainty_estimation/calculate_uncertainty.py`` (``uncertainty-nlp-2``).

Paired tests need to know which uncertainty belongs to which item, but the old
pipeline saves ``{uncertainty: [[is_wrong_net0, ...], ...]}``: items are grouped
by uncertainty and their ids are dropped. Rather than re-implement each method,
this script runs the *unchanged* old ``calculate_uncertainties`` with a tagged
mistakes dict whose entries are the item ids. The old functions copy those
entries verbatim into their output, so every uncertainty comes back attached to
its item. The real mistakes are rebuilt exactly as ``get_classifier_mistakes``
builds them (without its side effect of writing a JSON).

Usage (from anywhere):
    python -m stats.collect_ue_items \
        --old-repo /path/to/uncertainty-nlp-2 \
        --config /path/to/uncertainty-nlp-2/uncertainty_estimation/config_UE.json \
        --methods "TV_pred_exps_p(1-p)_sum_CE_var" "TV_CE_p(1-p)_sum_CE_var" \
                  "TV_gt_exps_p(1-p)_sum_CE_var" mcmc \
        --out results/paired_tests/pmqa_items.npz \
        --validate-dir /path/to/uncertainty-nlp-2/uncertainty_estimation/results/bert_v2/uncertainty
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from stats.data import ItemLevelData
from stats.sample_curves import make_grid, net_curves


OUTPUT_KEYS = ("path2outputs", "path2densities")


def _int_net_keys(outputs: dict) -> dict:
    """JSON stores net ids as strings; the old code indexes pickled outputs by int (``outputs[0]``)."""
    if not isinstance(outputs, dict):
        raise TypeError(f"Expected outputs as {{net_id: {{item_id: ...}}}}, got {type(outputs).__name__}.")
    return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in outputs.items()}


def pickle_json_outputs(config: dict, out_dir: Path) -> dict:
    """Copy of ``config`` in which model outputs saved as ``.json`` point to equivalent ``.pkl`` files.

    The old pipeline reads every model output with ``load_pickle`` and finds MCMC
    samples with ``glob("*.pkl")``; BloodyWell outputs were saved as JSON. Each
    ``path2outputs*`` / ``path2densities`` value that is a ``.json`` file, or a
    directory holding ``.json`` but no ``.pkl`` files, is converted (file stems
    are kept: ``calc_mcmc`` uses them as sample names). The old code is unchanged.
    """
    config = copy.deepcopy(config)
    converted: dict[Path, str] = {}

    def to_pickle(src: Path, dst: Path) -> None:
        with open(src, "r", encoding="utf-8") as f:
            outputs = _int_net_keys(json.load(f))
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump(outputs, f)

    def convert(value: str) -> str:
        path = Path(value).resolve()
        if path in converted:
            return converted[path]
        if path.is_file() and path.suffix == ".json":
            target = out_dir / f"{len(converted)}" / f"{path.stem}.pkl"
            to_pickle(path, target)
        elif path.is_dir() and not any(path.glob("*.pkl")) and any(path.glob("*.json")):
            target = out_dir / f"{len(converted)}_{path.name}"
            for src in sorted(path.glob("*.json")):
                to_pickle(src, target / f"{src.stem}.pkl")
        else:
            return value
        print(f"  {value}: JSON outputs converted to pickle for the old pipeline")
        converted[path] = str(target)
        return converted[path]

    def walk(node: dict) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, str) and key.startswith(OUTPUT_KEYS):
                node[key] = convert(value)

    walk(config)
    return config


def build_mistakes(config: dict, load_pickle, load_json) -> dict[int, dict]:
    """Base-ensemble mistakes, built exactly as the old ``get_classifier_mistakes``."""
    outputs = load_pickle(config["path2outputs_classifier"])
    test_markup = load_json(config["test_markup"])
    mistakes = {}
    for net_id, net_outputs in outputs.items():
        net_id = int(net_id)
        mistakes[net_id] = {}
        for pubid, info in net_outputs.items():
            if pubid in test_markup:
                mistakes[net_id][pubid] = info["gt"] != info["pred"]
    return mistakes


def method_files(save_dir: Path, method: str, info: dict, n_nets: int) -> list[Path]:
    """Where the old pipeline writes a method's uncertainties (as plot_rejection_curves reads them)."""
    template = info.get("single_net_file_name_template")
    if template is None:
        return [save_dir / f"{method}.json"]
    return [save_dir / method / f"{template}_{net}.json" for net in range(n_nets)]


def read_tagged(path: Path) -> dict[str, str]:
    """``{item_id: uncertainty_str}`` from an output file produced with tagged mistakes."""
    with open(path, "r", encoding="utf-8") as f:
        grouped = json.load(f)
    item_to_unc = {}
    for unc, entries in grouped.items():
        for tags in entries:
            item = str(tags[0])
            if item in item_to_unc:
                raise ValueError(
                    f"{path.name}: item {item} appears more than once. The old function that "
                    "wrote this file accumulates items across nets (e.g. calc_huq / calc_ddu "
                    "keep one dict for all nets), so it has no single uncertainty per item."
                )
            item_to_unc[item] = unc
    return item_to_unc


def validate(method: str, tagged: list[dict[str, str]], item_ids: list[str], wrong: np.ndarray,
             mistake_vectors: dict[str, tuple], validate_dir: Path, info: dict) -> None:
    """Check the recomputed uncertainties against the files saved by the original run.

    Exact string equality is reported but not required: float32 reductions can
    differ in the last bits between numpy builds. What must agree is the
    rejection curve, which depends only on the uncertainty ranking.
    """
    saved_files = method_files(validate_dir, method, info, len(tagged))
    grid = make_grid(1000)
    exact, worst = True, 0.0
    for net, (tagged_map, saved_path) in enumerate(zip(tagged, saved_files)):
        with open(saved_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        expected = Counter((unc, tuple(bool(v) for v in vec))
                           for unc, vecs in saved.items() for vec in vecs)
        got = Counter((unc, mistake_vectors[item]) for item, unc in tagged_map.items())
        exact &= expected == got

        # One file per net -> that net's curve; a single file -> all nets' curves.
        nets = [net] if len(saved_files) > 1 else list(range(wrong.shape[0]))
        saved_unc = np.array([float(unc) for unc, vecs in saved.items() for _ in vecs])
        saved_wrong = np.array([vec for vecs in saved.values() for vec in vecs], dtype=bool).T
        recomputed = np.array([float(tagged_map[item]) for item in item_ids])
        worst = max(worst, np.abs(net_curves(saved_unc, saved_wrong[nets], grid)
                                  - net_curves(recomputed, wrong[nets], grid)).max())
    if worst > 1e-9:
        raise AssertionError(f"{method}: rejection curves differ from the saved run "
                             f"(max |diff| = {worst:.2e}).")
    detail = "identical uncertainties" if exact else "same curves, last-bit float differences"
    print(f"  {method}: matches the saved run ({detail}; max curve diff {worst:.1e})")


def method_precisions(config: dict, methods: list[str], n_nets: int,
                      validate_dir: Path | None, overrides: dict[str, int]) -> dict[str, int]:
    """``sign_figs_num`` to use per method.

    Taken from, in order: ``overrides``; the saved files of the original run
    (number of decimals in their keys), if ``validate_dir`` is given; the config.
    """
    precision = {}
    for method in methods:
        figs = config["sign_figs_num"]
        if validate_dir is not None:
            saved = method_files(validate_dir, method, config["UE_methods"][method], n_nets)[0]
            with open(saved, "r", encoding="utf-8") as f:
                key = next(iter(json.load(f)))
            figs = len(key.split(".")[1]) if "." in key else 0
        precision[method] = overrides.get(method, figs)
    return precision


def collect(old_repo: Path, module: str, config_path: Path, methods: list[str] | None,
            validate_dir: Path | None, sign_figs: dict[str, int] | None = None) -> ItemLevelData:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Relative paths in old configs are relative to the old repo root.
    sys.path.insert(0, str(old_repo))
    os.chdir(old_repo)
    old = importlib.import_module(module)

    methods = methods or list(config["UE_methods"])
    missing = [m for m in methods if m not in config["UE_methods"]]
    if missing:
        raise KeyError(f"Methods not in the config's UE_methods: {missing}")

    with tempfile.TemporaryDirectory() as tmp:
        config = pickle_json_outputs(config, Path(tmp) / "outputs_pkl")
        mistakes = build_mistakes(config, old.load_pickle, old.load_json)
        net_ids = list(mistakes)
        if net_ids != list(range(len(net_ids))):
            raise ValueError(f"Expected base nets 0..n-1 in order, got {net_ids}.")
        item_ids = [str(item) for item in mistakes[0]]
        wrong = np.array([[mistakes[net][item] for item in mistakes[0]] for net in net_ids])
        mistake_vectors = {str(item): tuple(bool(mistakes[net][item]) for net in net_ids)
                           for item in mistakes[0]}

        tagged_mistakes = {net: {item: item for item in per_net} for net, per_net in mistakes.items()}
        precision = method_precisions(config, methods, len(net_ids), validate_dir, sign_figs or {})

        data = ItemLevelData(item_ids=np.array(item_ids), wrong=wrong)
        # The old run may have used different precisions per method, and the
        # precision decides which uncertainties tie -- run once per precision.
        for figs in sorted(set(precision.values())):
            group = [m for m in methods if precision[m] == figs]
            run_config = copy.deepcopy(config)
            run_config["UE_methods"] = {m: config["UE_methods"][m] for m in group}
            run_config["sign_figs_num"] = figs
            run_config["path2save_uncertainties"] = str(Path(tmp) / f"figs_{figs}")
            print(f"Running the old calculate_uncertainties (sign_figs_num={figs}) for: "
                  f"{', '.join(group)}")
            old.calculate_uncertainties(run_config, tagged_mistakes)

        for method in methods:
            info = config["UE_methods"][method]
            files = method_files(Path(tmp) / f"figs_{precision[method]}", method, info, len(net_ids))
            tagged = [read_tagged(path) for path in files]
            for tagged_map in tagged:
                if set(tagged_map) != set(item_ids):
                    raise ValueError(f"{method}: output does not cover exactly the test items.")
            per_file = [np.array([float(tagged_map[item]) for item in item_ids]) for tagged_map in tagged]
            data.uncertainty[method] = per_file[0] if len(per_file) == 1 else np.stack(per_file)
            if validate_dir is not None:
                validate(method, tagged, item_ids, wrong, mistake_vectors, validate_dir, info)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old-repo", type=Path, required=True,
                        help="Root of the old repo (the directory containing uncertainty_estimation/).")
    parser.add_argument("--module", default="uncertainty_estimation.calculate_uncertainty",
                        help="Import path of the old calculate_uncertainty module.")
    parser.add_argument("--config", type=Path, required=True, help="The old config_UE.json.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="UE_methods to collect (default: all in the config).")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--validate-dir", type=Path, default=None,
                        help="Saved 'uncertainty' dir of the original run. Each method is then "
                             "recomputed at the precision it was saved with and checked against it.")
    parser.add_argument("--sign-figs", nargs="*", default=[], metavar="METHOD=N",
                        help="Per-method sign_figs_num override, e.g. mcmc=3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    config = args.config.resolve()
    validate_dir = args.validate_dir.resolve() if args.validate_dir else None
    sign_figs = {m: int(n) for m, n in (item.rsplit("=", 1) for item in args.sign_figs)}
    data = collect(args.old_repo.resolve(), args.module, config, args.methods, validate_dir, sign_figs)
    data.save(out)
    print(f"Saved {len(data.item_ids)} items x {data.wrong.shape[0]} nets, "
          f"{len(data.uncertainty)} method(s) to {out}")


if __name__ == "__main__":
    main()

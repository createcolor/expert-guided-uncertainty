# Segmentation configs

Two kinds of config live here:

- **Train / inference configs** (`train_*.json`, `test_config.json`) — drive
  `train.py` / `inference.py`.
- **Rejection configs** (`rejection_curve_ens.json`, `rejection_configs_lidc/*`)
  — drive the `uncertainty.py` → `rejection_curves.py` pipeline.

## The rejection pipeline shares one config

`uncertainty.py` and `rejection_curves.py` read the **same** config file; each
uses the subset of fields it needs.

```bash
# 1. predictions.pt -> per-pixel uncertainty map (uncertainty.pt)
python tasks/segmentation/uncertainty.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/tv_cae.json \
  -o results/segmentation/lidc/

# 2. uncertainty.pt + predictions + labels -> rejection_curve_<c>.json
python tasks/segmentation/rejection_curves.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/tv_cae.json \
  -u results/segmentation/lidc/uncertainty.pt \
  -s results/segmentation/lidc/
```

## Field reference

| Field                | Used by        | Meaning |
|----------------------|----------------|---------|
| `models_predictions` | both           | `predictions.pt` from the model ensemble (epistemic source). |
| `models_ids`         | both           | Int (count) or list of ensemble member indices to use. |
| `models_method`      | uncertainty    | Epistemic estimator: `"var"` (or `"none"`). |
| `expert_predictions` | uncertainty    | Aleatoric source; **omit for epistemic-only.** Path to a `.pt` (plain / abnn) or a checkpoint directory (mcmc). |
| `expert_ids`         | uncertainty    | Int or list of expert/member indices. |
| `expert_method`      | uncertainty    | Aleatoric estimator: `"bern_var"`, `"mcmc"`, `"abnn"`, or `"none"`. |
| `step_size`, `total_epochs` | uncertainty | MCMC only: checkpoint stride and final epoch for `load_mcmc`. |
| `dataset`            | rejection      | `"lidc"` or `"riga"` (selects the label loader). |
| `data_path`          | rejection      | Path to the dataset (for ground-truth labels). |
| `expert_id`          | rejection      | Which expert's annotation to score Dice against. |

## Method variants (LIDC examples)

| Config                     | Uncertainty | Notes |
|----------------------------|-------------|-------|
| `prediction_variance.json` | epistemic only (no expert source) | model-ensemble variance. |
| `tv_base.json`             | total variance, aleatoric from the model ensemble itself | `expert_predictions` = model preds, `bern_var`. |
| `tv_cae.json`              | total variance, CAE (proposed) | `expert_predictions` = CAE ensemble. |
| `tv_experts.json`          | total variance, ground-truth experts | `expert_predictions` = expert annotations. |
| `mcmc.json`                | MCMC baseline | aleatoric from an MCMC chain (`step_size`/`total_epochs`). |
| `abnn.json`                | ABNN baseline | aleatoric from ABNN samples. |

`rejection_curve_ens.json` is the multiclass RIGA equivalent of `tv_base`.
Binary (LIDC) and multiclass (RIGA) use the same scripts — the number of
evaluated classes is inferred from the predictions.

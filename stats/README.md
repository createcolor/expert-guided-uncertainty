# Paired significance tests

Replaces the Friedman / Nemenyi critical-difference analysis. That analysis
treated every throwaway rate of every rejection curve as an independent
"dataset" (16 rates × 4 tasks = 64 rows), but points on one curve, and curves of
ensemble members scored on the same test set, are strongly correlated, which
inflates significance.

Here the **independent units are the test items**: questions (PubMedQA), wells
(BloodyWell) and images (LIDC-IDRI, RIGA). For each dataset we compare our
method, *Total variance, CAE*, with three references: *Total variance, base*
(strong baseline), a SotA method (*Entropy and MI* for BloodyWell and PubMedQA,
*MCMC* for LIDC-IDRI and RIGA, as reported in the paper's Tables 1 and 2) and
*Total variance, experts* (expert method).

## What is reported

For each comparison:

- **ΔAAC = AAC(ours) − AAC(other)** on the full test set (negative = ours better).
  AAC is computed exactly as for the paper (eq. 5): the area between the oracle
  curve and the base ensemble's rejection curve averaged over its nets, over
  throwaway rates `[0, 0.8]`. For PubMedQA / BloodyWell the oracle is that of a
  model with the ensemble's mean error rate; for segmentation it is the saved
  oracle run. The oracle term is the same for all methods on a resample, so it
  does not change ΔAAC.
- **95% CI and p-value from a paired bootstrap over test items.** Items are
  resampled with replacement and *both* methods' AAC are recomputed on the same
  resample. The p-value tests ΔAAC = 0 with the centred bootstrap distribution.
- **Wilcoxon signed-rank p-value on per-image ΔAAC** (segmentation only). There,
  dataset AAC is a mean over images, so per-image differences are valid paired
  observations.
- **Holm-adjusted p-value** over all comparisons in the run (the family).

## Running on the server

The collectors import the old code, so run them in the old repos' environment.
`run_paired_tests` only needs numpy and scipy. Run everything from the root of
this repo; outputs go to `results/paired_tests/` (git-ignored).

```bash
# PubMedQA: per-item uncertainties from the old uncertainty-nlp-2 pipeline
python -m stats.collect_ue_items --old-repo <uncertainty-nlp-2> \
  --config <uncertainty-nlp-2>/uncertainty_estimation/config_UE.json \
  --methods "TV_pred_exps_p(1-p)_sum_CE_var" "TV_CE_p(1-p)_sum_CE_var" \
            "TV_gt_exps_p(1-p)_sum_CE_var" entropy_CE_sum_CE \
  --validate-dir <uncertainty-nlp-2>/uncertainty_estimation/results/bert_v2/uncertainty \
  --out results/paired_tests/pmqa_items.npz

# BloodyWell: saved per-well uncertainties + the base nets' mistakes (no recomputation).
# --misclass is <nets>_misclass.json from the classification inference.py; or pass
# --outputs <nets>_<ep>ep_outputs.json to rebuild the mistakes from the outputs.
# The saved CAE curve is from another run (see notes), so CAE is not validated
# against it; run_paired_tests checks its AAC against the paper's 2.18 instead.
# Entropy's file lacks some wells; --missing-lowest ranks them as the old code did.
python -m stats.collect_bloodywell \
  --test-markup tasks/classification/markup/BloodyWell/test_dataset_10_10_80.json \
  --misclass /alpha/korchagin/projects/expressto/nn/outputs/MobileNet_small_10_10_80_meta_8_misclass.json \
  --uncertainty-dir <uncertainty-nlp>/uncertainty_estimation/results/bloodywell \
  --methods "TV_pred_exps_p(1-p)_sum_CE_var" "TV_CE_p(1-p)_sum_CE_var" \
            "TV_gt_exps_p(1-p)_sum_CE_var" entropy \
  --missing-lowest entropy \
  --skip-validate "TV_pred_exps_p(1-p)_sum_CE_var" \
  --validate-dir <uncertainty-nlp-2>/uncertainty_estimation/results/bloodywell/rejection_curves \
  --out results/paired_tests/bloodywell_items.npz

# RIGA and LIDC-IDRI: per-image Dice-rejection areas from the old segmentation scripts
python -m stats.collect_segmentation --old-seg-dir <confidence-aware-uncertainty>/segmentation \
  --spec stats/configs/segmentation_riga.json \
  --validate-dir <confidence-aware-uncertainty>/segmentation/rcs/e3_10k_microavg_std_full_dice \
  --out results/paired_tests/riga_images.npz
python -m stats.collect_segmentation --old-seg-dir <confidence-aware-uncertainty>/segmentation \
  --spec stats/configs/segmentation_lidc.json \
  --validate-dir <confidence-aware-uncertainty>/segmentation/rcs/e1_lidc_10k_microavg_std_full_dice \
  --out results/paired_tests/lidc_images.npz

# All comparisons -> results/paired_tests/paired_tests.{md,csv,json}
python -m stats.run_paired_tests -c stats/configs/paired_tests.json -o results/paired_tests
```

## How the run guards against wrong numbers

1. **Collectors reproduce the original runs.** With `--validate-dir`, every
   method's curves are compared with the files the original run saved:
   uncertainty files for PubMedQA / BloodyWell, `dice_scores.npy` for
   segmentation. A mismatch aborts the collection.
2. **`run_paired_tests` reproduces the paper's AAC.** Each dataset in
   `configs/paired_tests.json` lists `expected_aac`. These are the AACs of the
   original runs, computed from their saved curves by `stats.expected_aac`
   (sources are noted next to each). Before any test is run, the full-test-set
   AAC of every method must match within `reproduction_rtol` / `reproduction_atol`;
   otherwise the run stops with `ReproductionError`. The AACs are then compared
   with `paper_aac`, the values printed in the paper's Tables 1 and 2; a
   difference above `paper_atol` (one unit in the last printed digit) is
   reported as a warning.
3. **Unit tests** (`tests/test_paired_stats.py`, `tests/test_paired_pipeline.py`)
   check that the fast curve engines match verbatim copies of the old functions.
   They also check that the expected-AAC path (saved artifacts) and the observed
   path (collected data) agree, and that the runner refuses to report on a
   mismatch.

Expected AAC (×10⁻³) recorded in the config, with the paper's value in brackets
where it differs in the second decimal (all within one unit of that digit):

| Dataset | CAE (ours) | TV base | SotA | TV experts |
|---|---:|---:|---:|---:|
| BloodyWell | 2.18 | 4.28 | 4.00 (Entropy and MI) | 2.36 (2.37) |
| LIDC-IDRI | 2.44 | 2.63 | 3.96 (MCMC) | 1.38 |
| RIGA | 6.45 | 12.53 (12.54) | 27.12 (MCMC) | 6.27 |
| PubMedQA | 100.85 | 110.32 | 111.05 (Entropy and MI) | 66.11 |

PubMedQA and BloodyWell were reproduced locally from collected data.

## Notes on the old pipelines

- **How items are recovered.** The old QA pipeline saves uncertainties grouped
  by value, without item ids. `collect_ue_items` runs the unchanged
  `calculate_uncertainties` with mistake entries replaced by item ids, which the
  old functions copy through verbatim. HUQ / DDU cannot be recovered this way:
  `calc_huq` / `calc_ddu` accumulate items across nets in one dict, so the
  collector rejects them with an explicit error.
- **BloodyWell** did not go through the QA pipeline. Its curves come from the
  classification `rejection_curves.py`, which reads one `{uncertainty: [well
  names]}` file per method and a `{net_id: [misclassified wells]}` file. Those
  already identify every well, so `collect_bloodywell` only reads them. The
  saved curves for `TV_CE`, `TV_gt_exps` and `CE` reproduce exactly with ties at
  full precision (the default); `experts_p(1-p)` needs `--round-digits
  "experts_p(1-p)=10"`. **CAE:** the saved CAE curve in `uncertainty-nlp-2`
  (AAC 2.22) is from a different run than the paper's. The CAE file
  `uncertainty-nlp/.../results/bloodywell/unc_TV_pred_exps_p(1-p)_sum_CE_var.json`
  gives 2.18, the paper's Table 1 value. A second copy on the server
  (`/alpha/projects/expressto/results/uncertainty/`) differs by ≤ 1.2e-4 per
  well and also rounds to 2.18. The paper's BloodyWell ± values are larger than
  the per-net std for every method (e.g. 0.33 vs 0.30 for CAE); PubMedQA's match.
  `stats.scan_bloodywell_candidates` ranks candidate files against a saved
  curve. **Entropy and MI:** `unc_entropy.json` lacks 514 test wells, all
  classified correctly by every net (their entropies presumably collided as dict
  keys). The old code never discards wells missing from the file, so
  `--missing-lowest entropy` ranks them as least uncertain; this reproduces the
  saved curve exactly. One well has a `NaN` entropy, which the curve engine ranks
  as most uncertain, also as in the saved curve. MCMC (not used for BloodyWell):
  per-net methods need `unc_<method>/*_<net>.json`
  for nets 0..9. The local `unc_mcmc/` holds `mcmc_400 … mcmc_685` instead, so
  the per-net MCMC files used for the paper (`results/mcmc_scores/mcmc_<net>.json`
  in `rejection_curves_args.json`) have to be found on the server.
- **Precision per method.** The original PubMedQA run used `sign_figs_num=20`
  for the TV methods but 3 for MCMC (more ties). With `--validate-dir` the
  precision is read from the saved files; without it, pass `--sign-figs mcmc=3`.
- **PubMedQA MCMC** is not reported in the paper (Table 1 shows "–"), so it is
  not a PubMedQA comparator. (The saved run's MCMC was over linear-probing heads.)
- **Segmentation** reuses the old `utils` package and copies each script's
  uncertainty dispatch verbatim. Per-method `final_activation` mirrors the `-f`
  flags in `plot_curves.sh` / `plot_curves_lidc.sh`. The old RIGA loader builds
  mask paths with `"/Image/"` and orders files by `iterdir()`, so it must run on
  the same (Linux) filesystem as the original.
- **LIDC rate grid.** The local copy of the old `rejection_curves.py` uses 1,000
  rates, but the saved LIDC results have 10,001 points. The spec therefore
  assumes `linspace(0.0001, 1, 10000)`, as for RIGA; `--validate-dir` confirms
  this or reports the mismatch.

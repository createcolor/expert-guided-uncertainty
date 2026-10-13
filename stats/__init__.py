"""Paired significance tests between uncertainty methods.

Replaces the Friedman/Nemenyi critical-difference analysis, which treated the
points of each rejection curve (throwaway rates) as independent datasets. Here
the independent units are the test items themselves -- questions (PubMedQA),
wells (BloodyWell) or images (LIDC-IDRI, RIGA) -- and every comparison is
paired: both methods are evaluated on the same resampled items.

Modules:
  - ``paired_stats``         paired bootstrap, Wilcoxon, Holm correction
  - ``sample_curves``        accuracy-rejection curves / AAC for item-level tasks
  - ``segmentation_curves``  per-image Dice-rejection curves for segmentation
  - ``data``                 the on-disk formats produced by the collectors
  - ``collect_ue_items``     PubMedQA / BloodyWell data from the old pipeline
  - ``collect_segmentation`` LIDC-IDRI / RIGA data from the old pipeline
  - ``run_paired_tests``     runs all comparisons and writes the results table
"""

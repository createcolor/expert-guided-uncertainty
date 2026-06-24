# Expert-guided Uncertainty Modeling

Code for the paper **"Enhancing the Reliability of Medical AI through Expert-guided Uncertainty Modeling"** (Khalin, Zaychenkova, Yugay et al.).

---

## Overview

AI systems in healthcare need reliable uncertainty estimates so that human experts can focus review on high-risk cases. Standard methods struggle with **aleatoric uncertainty** — inherent ambiguity in the data — because they lack a reliable ground truth for it.

We leverage **aggregated expert annotations** as a proxy for aleatoric uncertainty. These annotations are used to fine-tune a second ensemble (the **Confidence-Aware Ensemble, CAE**) on "soft" labels, while the base ensemble captures epistemic uncertainty. Total uncertainty follows from the **law of total variance**:

> TU(y) = EU(y) + AU(y) = Var(E[y|x,θ]) + E[Var(y|x,θ)]

**Key results across four medical tasks:**

| Task | Dataset | Metric | Improvement over best baseline |
|------|---------|--------|-------------------------------|
| Image classification | BloodyWell | AAC ↓ | −57% (Total variance, CAE) |
| Binary segmentation | LIDC-IDRI | AAC ↓ | −70% |
| Multiclass segmentation | RIGA | AAC ↓ | −79% |
| Multiple-choice QA | PubMedQA | AAC ↓ | −14% |

All improvements over *Total variance, base* (no expert knowledge). The proposed method is also statistically significantly better than all baselines by the Nemenyi test (α = 0.05).

---

## Repository structure

```
expert-guided-uncertainty/
├── tasks/
│   ├── classification/        # BloodyWell blood-typing classification
│   │   ├── train.py           # Train base / CAE ensemble
│   │   ├── inference.py       # Run ensemble, compute outputs
│   │   ├── rejection_curves.py  # Sample-level rejection (accuracy vs throwaway rate)
│   │   ├── models/MobileNet.py
│   │   ├── dataset.py
│   │   ├── utils.py
│   │   ├── sgld_optimizer.py
│   │   └── configs/
│   ├── segmentation/          # LIDC-IDRI (binary) + RIGA (multiclass)
│   │   ├── train.py           # Train UNet ensemble
│   │   ├── inference.py       # Run segmentation inference
│   │   ├── rejection_curves.py       # Binary: pixel-level rejection
│   │   ├── rejection_curves_multiclass.py  # Multiclass: pixel-level rejection
│   │   ├── models/            # UNet, UNet_ABNN, BNL
│   │   ├── utils/             # Dataset, eval, SGLD, uncertainty
│   │   └── configs/
│   └── qa/                    # PubMedQA question answering
│       ├── bert/              # BERT fine-tuning (training from PMQA; baselines from nlp-2)
│       │   ├── finetune_no_reasoning.py
│       │   ├── finetune_reasoning_required.py
│       │   ├── pretrain_no_reasoning.py
│       │   ├── pretrain_reasoning_required.py
│       │   ├── train_ensemble_expert_average.py
│       │   ├── train_ensemble_final_decision.py
│       │   ├── train_single.py
│       │   ├── inference.py
│       │   ├── inference_ddu.py       # DDU baseline
│       │   ├── inference_dropout.py   # MC Dropout baseline
│       │   └── inference_test_augs.py # Test-time augmentation baseline
│       ├── llm/               # Mistral inference for uncertainty scores
│       │   ├── inference_mistral.py
│       │   ├── extract_embeds.py
│       │   └── dataset.py
│       └── rejection_curves.py  # Sample-level rejection for QA
│
├── uncertainty/               # Shared, task-agnostic modules
│   ├── metrics.py             # compute_aac(), compute_auroc() — used by all tasks
│   ├── variance_uncertainty.py  # TV decomposition for image tasks
│   ├── calculate_uncertainty.py # All QA uncertainty methods (ensemble, MCMC, DDU, HUQ, …)
│   ├── utils.py
│   └── baselines/             # DDU, HUQ, MCDropout wrappers for classification
│
├── figures/                   # Scripts to reproduce paper figures
│   ├── plot_rejection_curves_qa.py   # Fig. 4d
│   ├── critical_diff_diag.ipynb      # Fig. 3 (Nemenyi diagram)
│   ├── error_auroc.py                # Table 3
│   ├── ensemble_size_heatmap.py      # Ablation: ensemble size
│   ├── analyze_experts.py
│   └── plot_expert_size.py
│
├── tests/
│   ├── test_metrics.py                       # AAC, AUROC
│   ├── test_variance_uncertainty.py          # TV decomposition
│   ├── test_classification_rejection_curves.py
│   └── test_qa_rejection_curves.py
│
├── conftest.py
└── requirements.txt
```

---

## Data layout

All scripts and configs use paths **relative to the repository root**. Place your data under:

```
expert-guided-uncertainty/
└── data/
    ├── BloodyWell/          # BloodyWell image dataset (classification)
    ├── RIGA/                # RIGA multiclass segmentation dataset
    │   └── DiscRegion/
    ├── LIDC/                # LIDC-IDRI binary segmentation dataset
    │   ├── train_data.pickle
    │   ├── test_data.pickle
    │   └── experts_lidc.pt  # Aggregated expert annotations
    ├── nets/                # Trained model checkpoints
    │   └── <net_name>/
    ├── predictions/         # Saved inference outputs (segmentation)
    │   ├── LIDC/
    │   │   ├── e1/          # Base ensemble predictions
    │   │   └── e1_cae/      # CAE predictions
    │   └── RIGA/
    │       └── e3/
    ├── bert/                # BERT model checkpoints (QA)
    │   └── ensemble_final_decision_v2/
    └── markup/              # Additional markup files (experts, MCMC splits)
```

Run all scripts from the repository root so that relative paths resolve correctly.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** The LLM fine-tuning path (`tasks/qa/llm/`) additionally requires [unsloth](https://github.com/unslothai/unsloth) and a GPU with at least 24 GB VRAM.

---

## Reproducing results

All model training requires substantial compute and should be run on your own hardware. The steps below describe the full pipeline; inference and figure scripts can be run after loading pre-trained model outputs.

### 1. Image classification (BloodyWell)

**Train base ensemble:**
```bash
python tasks/classification/train.py \
  -n MobileNet_base \
  -c tasks/classification/configs/train_configs/<config>.json \
  -m tasks/classification/markup/BloodyWell/train_dataset_10_10_80.json
```

**Run inference and save outputs:**
```bash
python tasks/classification/inference.py \
  -ctest tasks/classification/configs/test_configs/<config>.json \
  -o outputs/classification/
```

**Compute rejection curves:**
```bash
python tasks/classification/rejection_curves.py \
  -tm tasks/classification/markup/BloodyWell/test_dataset_10_10_80.json \
  -i tasks/classification/configs/rejection_curves_args.json \
  -ds results/classification/
```

### 2. Segmentation (LIDC-IDRI / RIGA)

**Train UNet ensemble (RIGA, standard cross-entropy):**
```bash
python tasks/segmentation/train.py -c tasks/segmentation/configs/train_ensemble.json
```

**Train UNet ensemble with EDL loss:**
```bash
# RIGA (multiclass)
python tasks/segmentation/train.py -c tasks/segmentation/configs/train_ensemble_edl_riga.json

# LIDC-IDRI (binary)
python tasks/segmentation/train.py -c tasks/segmentation/configs/train_ensemble_edl_lidc.json
```

**Run inference and save predictions:**
```bash
python tasks/segmentation/inference.py -c tasks/segmentation/configs/test_config.json
```

**Compute pixel-level rejection curves (binary, LIDC-IDRI):**
```bash
# Total variance — base ensemble (epistemic only)
python tasks/segmentation/rejection_curves.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/tv_base.json \
  -s results/segmentation/binary/

# Total variance — CAE (epistemic + aleatoric)
python tasks/segmentation/rejection_curves.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/tv_cae.json \
  -s results/segmentation/binary/

# Total variance — oracle expert annotations
python tasks/segmentation/rejection_curves.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/tv_experts.json \
  -s results/segmentation/binary/

# Prediction variance baseline
python tasks/segmentation/rejection_curves.py \
  -c tasks/segmentation/configs/rejection_configs_lidc/prediction_variance.json \
  -s results/segmentation/binary/
```

**Multiclass rejection curves (RIGA):**
```bash
python tasks/segmentation/rejection_curves_multiclass.py \
  -c tasks/segmentation/configs/rejection_curve_ens.json \
  -s results/segmentation/multiclass/
```

### 3. QA — BERT (PubMedQA)

**Pre-train on unlabelled data:**
```bash
python tasks/qa/bert/pretrain_no_reasoning.py \
  -c tasks/qa/bert/pretrain_artificial_config_no_reasoning.json
```

**Fine-tune base ensemble:**
```bash
python tasks/qa/bert/finetune_no_reasoning.py \
  -c tasks/qa/bert/finetune_labelled_only_config.json
```

**Fine-tune CAE on expert soft labels:**
```bash
python tasks/qa/bert/train_ensemble_expert_average.py
```

**Run inference (ensemble):**
```bash
python tasks/qa/bert/inference.py -c tasks/qa/bert/inference.json
```

**Compute uncertainty and rejection curves:**
```bash
python uncertainty/calculate_uncertainty.py -c config_UE.json
python tasks/qa/rejection_curves.py
```

### 4. QA — LLM (Mistral)

**Run Mistral inference to collect answer and confidence scores:**
```bash
python tasks/qa/llm/inference_mistral.py
```

**Extract embeddings:**
```bash
python tasks/qa/llm/extract_embeds.py
```

---

## Reproducing figures

After running inference and saving model outputs, the figures can be generated without additional compute:

```bash
# Fig. 4 rejection curves (QA)
python figures/plot_rejection_curves_qa.py

# Fig. 3 Nemenyi critical difference diagram
jupyter notebook figures/critical_diff_diag.ipynb

# Table 3 error AUROC
python figures/error_auroc.py

# Ablation: ensemble size
python figures/ensemble_size_heatmap.py
```

---

## Running tests

```bash
pytest tests/
```

Tests cover the task-agnostic metrics (`compute_aac`, `compute_auroc`), the variance-based uncertainty decomposition, and the rejection curve logic for both classification/segmentation (sample-level) and QA (sample-level, different data format). They do **not** require GPU or trained models.

---

## Citation

```bibtex
@article{khalin2025expertguided,
  title   = {Enhancing the Reliability of Medical {AI} through Expert-guided Uncertainty Modeling},
  author  = {Khalin, Aleksei and Zaychenkova, Ekaterina and Yugay, Aleksandr and
             Goncharov, Andrey and Korchagin, Sergey and Zaytsev, Alexey and Ershov, Egor},
  year    = {2025}
}
```

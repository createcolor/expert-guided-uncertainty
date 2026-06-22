import sys
from pathlib import Path

from load_dataset import load_labelled_ds
from memory_sandbox import memory_sandbox_worker, run_in_memory_sandbox
from process_dataset import prepare_data_expert_average
from sklearn.model_selection import KFold
from train_single import train_on_PubMedQA

SEED = 42
MODEL_NAME = "answerdotai/ModernBERT-large"


@memory_sandbox_worker
def finetune_expert_average(prev_model_path, train_idx, test_idx, config_path, save_path):
    labelled_ds, _ = load_labelled_ds(test_train_split=False)

    train_ds = labelled_ds.select(train_idx)
    test_ds = labelled_ds.select(test_idx)

    return train_on_PubMedQA(train_ds, test_ds, prepare_data_expert_average, config_path, save_path, prev_model_path, problem_type="multi_label_classification")


def ensemble_train_on_PubMedQA(
    finetune_config_path: Path,
    pretrain_basedir: Path,
    save_basedir: Path,
    ensemble_size=10,
):
    labelled_ds, _ = load_labelled_ds(test_train_split=False)

    kf = KFold(n_splits=ensemble_size, shuffle=True, random_state=SEED)
    for fold_idx, (labelled_train_idx, labelled_test_idx) in enumerate(kf.split(labelled_ds)):
        print(f"Fold {fold_idx}/{ensemble_size - 1} -> start", finetune_config_path)

        pretrain_model_path = pretrain_basedir.joinpath(str(fold_idx)).joinpath("pretrain_reasoning_required")
        save_path = save_basedir.joinpath(str(fold_idx))

        run_in_memory_sandbox(
            finetune_expert_average,
            pretrain_model_path,
            labelled_train_idx,
            labelled_test_idx,
            finetune_config_path,
            save_path,
        )

        print(f"Fold {fold_idx}/{ensemble_size - 1} -> done", save_path)


if __name__ == "__main__":
    ensemble_train_on_PubMedQA(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

import sys
from pathlib import Path

from load_dataset import load_artificial_ds, load_labelled_ds
from memory_sandbox import memory_sandbox_worker, run_in_memory_sandbox
from process_dataset import prepare_data_no_reasoning, prepare_data_reasoning_required
from sklearn.model_selection import KFold
from train_single import train_on_PubMedQA

SEED = 42
MODEL_NAME = "answerdotai/ModernBERT-large"


@memory_sandbox_worker
def pretrain_no_reasoning(pretrain_size, fold_idx, config_path, save_path):
    train_ds, test_ds = load_artificial_ds(test_size=1 - pretrain_size, seed=SEED + fold_idx)

    return train_on_PubMedQA(train_ds, test_ds, prepare_data_no_reasoning, config_path, save_path, MODEL_NAME)


@memory_sandbox_worker
def pretrain_reasoning_required(prev_model_path, pretrain_size, fold_idx, config_path, save_path):
    train_ds, test_ds = load_artificial_ds(test_size=1 - pretrain_size, seed=SEED + fold_idx)

    return train_on_PubMedQA(
        train_ds, test_ds, prepare_data_reasoning_required, config_path, save_path, prev_model_path
    )


@memory_sandbox_worker
def finetune_reasoning_required(prev_model_path, train_idx, test_idx, config_path, save_path):
    labelled_ds, _ = load_labelled_ds(test_train_split=False)

    train_ds = labelled_ds.select(train_idx)
    test_ds = labelled_ds.select(test_idx)

    return train_on_PubMedQA(
        train_ds, test_ds, prepare_data_reasoning_required, config_path, save_path, prev_model_path
    )


def ensemble_train_on_PubMedQA(
    # pretrain config, finetune config
    config_paths: tuple[Path, Path],
    save_basedir: Path,
    ensemble_size=10,
    pretrain_size=0.7,
):
    pretrain_config_path, finetune_config_path = config_paths

    labelled_ds, _ = load_labelled_ds(test_train_split=False)

    kf = KFold(n_splits=ensemble_size, shuffle=True, random_state=SEED)
    for fold_idx, (labelled_train_idx, labelled_test_idx) in enumerate(kf.split(labelled_ds)):
        print(f"Fold {fold_idx}/{ensemble_size - 1} -> start", pretrain_config_path, finetune_config_path)

        member_save_basedir = save_basedir.joinpath(str(fold_idx))

        pretrain_no_reasoning_savepath = member_save_basedir.joinpath("pretrain_no_reasoning")
        print(f"Fold {fold_idx} -> pretrain_no_reasoning", pretrain_no_reasoning_savepath)
        run_in_memory_sandbox(
            pretrain_no_reasoning, pretrain_size, fold_idx, pretrain_config_path, pretrain_no_reasoning_savepath
        )

        pretrain_reasoning_required_savepath = member_save_basedir.joinpath("pretrain_reasoning_required")
        print(f"Fold {fold_idx} -> pretrain_reasoning_required", pretrain_reasoning_required_savepath)
        run_in_memory_sandbox(
            pretrain_reasoning_required,
            pretrain_no_reasoning_savepath,
            pretrain_size,
            fold_idx,
            pretrain_config_path,
            pretrain_reasoning_required_savepath,
        )

        finetune_reasoning_required_savepath = member_save_basedir.joinpath("finetune_reasoning_required")
        print(f"Fold {fold_idx} -> finetune_reasoning_required", finetune_reasoning_required_savepath)
        run_in_memory_sandbox(
            finetune_reasoning_required,
            pretrain_reasoning_required_savepath,
            labelled_train_idx,
            labelled_test_idx,
            finetune_config_path,
            finetune_reasoning_required_savepath,
        )

        print(f"Fold {fold_idx}/{ensemble_size - 1} -> done", finetune_reasoning_required_savepath)


if __name__ == "__main__":
    ensemble_train_on_PubMedQA((Path(sys.argv[1]), Path(sys.argv[2])), Path(sys.argv[3]))

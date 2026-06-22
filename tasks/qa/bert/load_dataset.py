import json
from pathlib import Path

from datasets import Dataset, load_dataset


def load_labelled_ds(test_train_split=True):
    ds: Dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

    if test_train_split:
        with open(
            Path(__file__).parent.joinpath("../../data/test_ground_truth.json").resolve(), "r", encoding="utf-8"
        ) as file:
            ground_truth_json = json.load(file)
        ground_truth_set = set(int(key) for key in ground_truth_json.keys())

        test_ds = ds.filter(lambda row: row["pubid"] in ground_truth_set)
        train_ds = ds.filter(lambda row: row["pubid"] not in ground_truth_set)

        return train_ds, test_ds

    return ds, ds


def load_artificial_ds(test_size=0.1, seed=42):
    ds: Dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")["train"]

    split = ds.train_test_split(test_size=test_size, seed=seed)
    test_ds = split["test"]
    train_ds = split["train"]

    return train_ds, test_ds

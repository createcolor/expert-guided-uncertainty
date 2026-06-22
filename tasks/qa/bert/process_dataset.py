import numpy as np
from datasets import Dataset
from datasets.formatting.formatting import LazyRow
from transformers import PreTrainedTokenizerFast

label2id = {"no": 0, "maybe": 1, "yes": 2}
id2label = {0: "no", 1: "maybe", 2: "yes"}
MATCHING_DICT = {"no": [1.0, 0.0, 0.0], "maybe": [0.0, 1.0, 0.0], "yes": [0.0, 0.0, 1.0]}


def prepare_data_reasoning_required(dataset: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    def process_row(row: LazyRow) -> LazyRow:
        text = f"{''.join(row['context']['contexts'])} {row['question']}"
        res = tokenizer(text, truncation=True)
        res["label"] = label2id[row["final_decision"]]
        return res

    dataset = dataset.map(process_row, num_proc=4)
    return dataset


def prepare_data_no_reasoning(dataset: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    def process_row(row: LazyRow) -> LazyRow:
        text = f"{''.join(row['context']['contexts'])} {row['long_answer']} {row['question']}"
        res = tokenizer(text, truncation=True)
        res["label"] = label2id[row["final_decision"]]
        return res

    dataset = dataset.map(process_row, num_proc=4)
    return dataset


def prepare_data_expert_average(dataset: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    def process_row(row: LazyRow) -> LazyRow:
        text = f"{''.join(row['context']['contexts'])} {row['question']}"
        res = tokenizer(text, truncation=True)

        reasoning_label = MATCHING_DICT["".join(row["context"]["reasoning_required_pred"])]
        free_label = MATCHING_DICT["".join(row["context"]["reasoning_free_pred"])]

        res["labels"] = np.array([reasoning_label, free_label]).mean(axis=0)
        return res

    dataset = dataset.map(process_row, num_proc=4)
    return dataset

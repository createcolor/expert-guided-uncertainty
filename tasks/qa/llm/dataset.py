from collections.abc import Callable
import torch
import pandas as pd
from datasets import Dataset
from datasets.formatting.formatting import LazyRow
from transformers import PreTrainedTokenizerFast


def chat_template_training(row: LazyRow, tokenizer: PreTrainedTokenizerFast) -> LazyRow:
    """
    Maps a row from the PubMedQA dataset to the json-like format supported by Qwen:
        {"role": <ROLE>, "content": <CONTENT>},
    where role is "system" for context, "user" for the question and "assistant" for the
    ground truth response.
    """
    row_json = [{"role": "system", "content": row["contexts"]},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["long_answer"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


def prepare_finetune_data(data: pd.DataFrame,
                          tokenizer: PreTrainedTokenizerFast,
                          test_size: float | None = 0.1) -> tuple[Dataset, Callable]:
    """
    Creates a datasets.Dataset instance from pandas data, applies a map, splits the data.
    """
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(lambda x: chat_template_training(x, tokenizer), num_proc=4)

    if test_size is not None:
        dataset = dataset.train_test_split(test_size=test_size)
    # Dynamic padding for efficiency

    def data_collator(batch):
        return tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)

    return dataset, data_collator


def chat_template_inference(row: LazyRow, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    """
    Maps a row from the PubMedQA dataset to the json-like format supported by Qwen:
        {"role": <ROLE>, "content": <CONTENT>},
    where role is "system" for context and "user" for the question.
    Tokenizes the input and return a torch.Tensor object to be fed into a model.
    """
    row_json = [{"role": "system", "content": row["contexts"]},
                {"role": "user", "content": row["question"]}]
    inputs = tokenizer.apply_chat_template(row_json, tokenize=True,
                                           add_generation_prompt=True, return_tensors='pt')
    return inputs

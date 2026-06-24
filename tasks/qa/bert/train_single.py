import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from tasks.qa.bert.process_dataset import id2label, label2id
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from utils import get_gpu_stats  # pylint: disable=import-error


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    correct = np.sum(preds == labels).item()
    total = labels.shape[0]
    accuracy = correct / total
    return {"accuracy": accuracy}

def compute_metrics_multi(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    correct = np.sum(np.isclose(probs, labels, atol=0.1))
    total = labels.shape[0] * labels.shape[1]
    accuracy = correct / total
    return {"accuracy": accuracy}


def train_on_PubMedQA(
    train_ds: Dataset,
    test_ds: Dataset,
    prepare_data,
    config_path: Path,
    save_path: Path,
    model_name_override: str | None = None,
    problem_type: str | None = None
):
    """
    Loads the model, the data and runs the training process.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    report_to = "none"
    if "wandb" in config:
        report_to = "wandb"
        wandb_config = config["wandb"]

        import wandb

        wandb.login()
        wandb.init(
            project="nlp-uncertainty",
            config=wandb_config,
        )

    seed = config["train_params"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model_name = model_name_override or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, **config["tokenizer_params"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **config["model_params"], num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    if problem_type is not None:
        model.config.problem_type = problem_type

    train_ds = prepare_data(train_ds, tokenizer)
    test_ds = prepare_data(test_ds, tokenizer)

    print(f"test_ds len {len(test_ds)}, train_ds len {len(train_ds)}")
    print(test_ds[:5]["pubid"])
    print(train_ds[:5]["pubid"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bf16 = False
    if torch.cuda.is_available():
        bf16 = True

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            **config["train_params"], output_dir=save_path, bf16=bf16, bf16_full_eval=bf16, report_to=report_to
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_multi if problem_type == 'multi_label_classification' else compute_metrics,
    )

    get_gpu_stats()
    trainer.train()
    trainer.save_model(save_path)

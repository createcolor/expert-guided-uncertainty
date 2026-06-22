import sys
from pathlib import Path

from load_dataset import load_labelled_ds
from process_dataset import prepare_data_no_reasoning
from tasks.qa.bert.train_single import train_on_PubMedQA

if __name__ == "__main__":
    train_ds, test_ds = load_labelled_ds()
    train_on_PubMedQA(train_ds, test_ds, prepare_data_no_reasoning, Path(sys.argv[1]), Path(sys.argv[2]))

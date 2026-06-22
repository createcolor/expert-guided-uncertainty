import sys
from pathlib import Path

from load_dataset import load_artificial_ds
from process_dataset import prepare_data_reasoning_required
from tasks.qa.bert.train_single import train_on_PubMedQA

if __name__ == "__main__":
    train_ds, test_ds = load_artificial_ds()
    train_on_PubMedQA(train_ds, test_ds, prepare_data_reasoning_required, Path(sys.argv[1]), Path(sys.argv[2]))



import gzip

from tqdm import tqdm

import prior

try:
    from prior import NoCacheLazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in [("train", 1000), ("val", 100)]:
        with gzip.open(f"{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = NoCacheLazyJsonDataset(
            data=houses, dataset="attr-onav-1k", split=split
        )
    return prior.DatasetDict(**data)

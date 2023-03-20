import gzip
from typing import Literal, Optional, Set

import prior

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")

VALID_SCENE_DATASETS = {"procthor-attr-onav"}


#%%
def load_dataset(
    minival: bool = False,
    scene_datasets: Optional[
        Set[Literal["procthor-attr-onav"]]
    ] = None,
) -> prior.DatasetDict:
    """Load the houses dataset."""
    assert isinstance(minival, bool)

    if scene_datasets is None:
        scene_datasets = VALID_SCENE_DATASETS
    else:
        for scene_dataset in scene_datasets:
            assert scene_dataset in VALID_SCENE_DATASETS
    scene_datasets = sorted(scene_datasets)

    data = {}
    for split in ["val"]:
        lines = []
        used_datasets = []
        for scene_dataset in scene_datasets:
            if split == "test" and scene_dataset == "robothor":
                continue

            with gzip.open(
                f"{scene_dataset}-locobot-{split}.jsonl.gz", "r"
            ) as f:
                lines.extend([line for line in f])
            used_datasets.append(scene_dataset)
        if split == "minival":
            split = "val"
        data[split] = LazyJsonDataset(
            data=lines,
            dataset="|".join(used_datasets) + "|attr-onav-eval",
            split=split,
        )

    return prior.DatasetDict(**data)

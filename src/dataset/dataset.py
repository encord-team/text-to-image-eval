from datasets import Dataset, DatasetDict, load_dataset

DATASETS = {"LungCancer4Types": "Kabil007/LungCancer4Types"}


class HFDataset:
    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in DATASETS:
            raise ValueError("Unsupported dataset")
        dataset_name = DATASETS[dataset_name]
        self.dataset = self._load_dataset(dataset_name)

    def _load_dataset(self, dataset_name) -> Dataset:
        try:
            dataset = load_dataset(dataset_name)
        except Exception as e:
            self.dataset = None
            raise ValueError(f"Failed to load dataset from HF {dataset_name}") from e
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]
        return dataset

from datasets import load_dataset


class HFDataset:
    def __init__(self, dataset_name) -> None:
        self.dataset = self._load_dataset(dataset_name)

    def _load_dataset(self, dataset_name):
        try:
            self.dataset = load_dataset(dataset_name)
        except Exception:
            self.dataset = None
            raise ValueError(f"Failed to load dataset from HF {dataset_name}")

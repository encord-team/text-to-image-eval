from datasets import Dataset, Split, load_dataset

DATASETS = {
    "LungCancer4Types": "Kabil007/LungCancer4Types",
    "plants": "sampath017/plants",
    "NIH-Chest-X-ray": "alkzar90/NIH-Chest-X-ray-dataset",
    "Alzheimer_MRI": "Falah/Alzheimer_MRI",
    "Skin_cancer": "marmal88/skin_cancer",
    "chest-xray-classification": "tpakov/chest-xray-classification",
    "sports-classification": "HES-XPLAIN/SportsImageClassification",
    "geo-landmarks": "Qdrant/google-landmark-geo",
}


class HFDataset:
    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unrecognized dataset: {dataset_name}")
        dataset_name = DATASETS[dataset_name]
        self.dataset = self._load_dataset(dataset_name)

    def _load_dataset(self, dataset_name) -> Dataset:
        try:
            dataset = load_dataset(dataset_name, split=Split.TRAIN)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: {dataset_name}") from e
        return dataset

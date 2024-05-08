from datasets import ClassLabel, DatasetDict, Sequence, Value, load_dataset
from datasets import Dataset as _RemoteHFDataset

from tti_eval.dataset import Dataset, Split


class HFDataset(Dataset):
    def __init__(
        self,
        title: str,
        *,
        split: Split = Split.ALL,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str | None = None,
        target_feature: str = "label",
        **kwargs,
    ):
        super().__init__(title, split=split, title_in_source=title_in_source, transform=transform, cache_dir=cache_dir)
        self._target_feature = target_feature
        self._setup(**kwargs)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)

    def set_transform(self, transform):
        super().set_transform(transform)
        self._dataset.set_transform(transform)

    @staticmethod
    def _get_available_splits(dataset_dict: DatasetDict) -> list[Split]:
        return [Split(s) for s in dataset_dict.keys() if s in [_ for _ in Split]] + [Split.ALL]

    def _get_hf_dataset_split(self, **kwargs) -> _RemoteHFDataset:
        try:
            if self.split == Split.ALL:  # Retrieve all the dataset data if the split is ALL
                return load_dataset(self.title_in_source, split="all", cache_dir=self._cache_dir.as_posix(), **kwargs)
            dataset_dict: DatasetDict = load_dataset(
                self.title_in_source,
                cache_dir=self._cache_dir.as_posix(),
                **kwargs,
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: `{self.title_in_source}`") from e

        available_splits = HFDataset._get_available_splits(dataset_dict)
        missing_splits = [s for s in Split if s not in available_splits]

        # Return target dataset if it already exists and won't be modified
        if self.split in [Split.VALIDATION, Split.TEST] and self.split in available_splits:
            return dataset_dict[self.split]
        if self.split == Split.TRAIN:
            if self.split in missing_splits:
                # Train split must always exist
                raise AttributeError(f"Missing train split in Hugging Face dataset: `{self.title_in_source}`")
            if not missing_splits:
                # No need to split the train dataset, can be returned as a whole
                return dataset_dict[self.split]

        # Get a 15% of the train dataset for each missing split (VALIDATION, TEST or both)
        # This operation includes data shuffling to prevent splits with skewed class counts because of the input order
        split_percent = 0.15 * len(missing_splits)
        split_seed = 42
        # Split the original train dataset into two, the final train dataset and the missing splits dataset
        split_to_dataset = dataset_dict["train"].train_test_split(test_size=split_percent, seed=split_seed)
        if self.split == Split.TRAIN:
            return split_to_dataset[self.split]

        if len(missing_splits) == 1:
            # One missing split (either VALIDATION or TEST), so we return the 15% stored in "test"
            return split_to_dataset["test"]
        else:
            # Both VALIDATION and TEST splits are missing
            # Each one will take a half of the 30% stored in "test"
            if self.split == Split.VALIDATION:
                return split_to_dataset["test"].train_test_split(test_size=0.5, seed=split_seed)["train"]
            else:
                return split_to_dataset["test"].train_test_split(test_size=0.5, seed=split_seed)["test"]

    def _setup(self, **kwargs):
        self._dataset = self._get_hf_dataset_split(**kwargs)

        if self._target_feature not in self._dataset.features:
            raise ValueError(f"The dataset `{self.title}` does not have the target feature `{self._target_feature}`")
        # Encode the target feature if necessary (e.g. use integer instead of string in the label values)
        if isinstance(self._dataset.features[self._target_feature], Value):
            self._dataset = self._dataset.class_encode_column(self._target_feature)

        # Rename the target feature to `label`
        # TODO do not rename the target feature but use it in the embeddings computations instead of the `label` tag
        if self._target_feature != "label":
            self._dataset = self._dataset.rename_column(self._target_feature, "label")

        label_feature = self._dataset.features["label"]
        if isinstance(label_feature, Sequence):  # Drop potential wrapper
            label_feature = label_feature.feature

        if isinstance(label_feature, ClassLabel):
            self.class_names = label_feature.names
        else:
            raise TypeError(f"Expected target feature of type `ClassLabel`, found `{type(label_feature).__name__}`")

    def __del__(self):
        for f in self._cache_dir.glob("**/*.lock"):
            f.unlink(missing_ok=True)

from datasets import ClassLabel, DatasetDict, Sequence, Value, load_dataset

from .base import Dataset, Split


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

    def _get_available_splits(self, **kwargs) -> list[Split]:
        datasets: DatasetDict = load_dataset(self.title_in_source, cache_dir=self._cache_dir.as_posix(), **kwargs)
        return list(Split(s) for s in datasets.keys() if s in [_ for _ in Split]) + [Split.ALL]

    def _setup(self, **kwargs):
        try:
            available_splits = self._get_available_splits(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: `{self.title_in_source}`") from e

        missing_splits = [s for s in Split if s not in available_splits]
        if self.split == Split.TRAIN and self.split in missing_splits:
            # Train split must always exist
            raise AttributeError(f"Missing train split in Hugging Face dataset `{self.title_in_source}`")

        # Select appropriate HF dataset split
        hf_split: str
        if self.split == Split.ALL:
            hf_split = str(self.split)
        elif self.split == Split.TEST:
            hf_split = f"train[{85}%:]" if self.split in missing_splits else str(self.split)
        elif self.split == Split.VALIDATION:
            # The range of the validation split in the training data may vary depending on whether
            # the test split is also missing
            hf_split = (
                f"train[{100 - 15 * len(missing_splits)}%:{100 - 15 * (len(missing_splits) - 1)}%]"
                if self.split in missing_splits
                else str(self.split)
            )
        elif self.split == Split.TRAIN:
            # Take into account the capacity taken by missing splits
            hf_split = f"train[:{100 - 15 * len(missing_splits)}%]"
        else:
            raise ValueError(f"Unhandled split type `{self.split}`")

        self._dataset = load_dataset(
            self.title_in_source,
            split=hf_split,
            cache_dir=self._cache_dir.as_posix(),
            **kwargs,
        )

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

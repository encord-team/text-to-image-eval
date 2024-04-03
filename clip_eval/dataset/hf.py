from datasets import ClassLabel, Sequence, Split, Value, load_dataset

from .base import Dataset


class HFDataset(Dataset):
    def __init__(
        self,
        title: str,
        *,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str | None = None,
        target_feature: str = "label",
        **kwargs,
    ):
        super().__init__(title, title_in_source=title_in_source, transform=transform, cache_dir=cache_dir)
        self._target_feature = target_feature
        self._setup(**kwargs)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)

    def set_transform(self, transform):
        super().set_transform(transform)
        self._dataset.set_transform(transform)

    def _setup(self, split: str | Split | None = None, **kwargs):
        try:
            # Retrieve the train data if no split has been explicitly specified
            if split is None:
                split = Split.TRAIN
            self._dataset = load_dataset(
                self.title_in_source,
                split=split,
                cache_dir=self._cache_dir.as_posix(),
                **kwargs,
            )

            if self._target_feature not in self._dataset.features:
                raise ValueError(
                    f"The dataset `{self.title}` does not have the target feature `{self._target_feature}`"
                )
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

        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: {self.title_in_source}") from e

    def __del__(self):
        for f in self._cache_dir.glob("**/*.lock"):
            f.unlink(missing_ok=True)

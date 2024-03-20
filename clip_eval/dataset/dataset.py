from abc import ABC, abstractmethod
from pathlib import Path

from datasets import ClassLabel, Sequence, Split, Value, load_dataset
from torch.utils.data import Dataset as TorchDataset

from clip_eval.constants import CACHE_PATH


class Dataset(TorchDataset, ABC):
    def __init__(
        self,
        title: str,
        *,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.transform = transform
        self.__title = title
        self.__title_in_source = title if title_in_source is None else title_in_source
        self.__class_names = []
        if cache_dir is None:
            cache_dir = CACHE_PATH
        self._cache_dir: Path = Path(cache_dir).expanduser().resolve() / "datasets" / title

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @property
    def title(self) -> str:
        return self.__title

    @property
    def title_in_source(self) -> str:
        return self.__title_in_source

    @property
    def class_names(self) -> list[str]:
        return self.__class_names

    @class_names.setter
    def class_names(self, class_names: list[str]) -> None:
        self.__class_names = class_names

    def set_transform(self, transform):
        self.transform = transform

    @property
    def text_queries(self) -> list[str]:
        return [f"An image of a {class_name}" for class_name in self.class_names]

    @abstractmethod
    def _setup(self, **kwargs):
        pass


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

from abc import ABC, abstractmethod
from pathlib import Path

from datasets import Split, load_dataset
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset, ABC):
    def __init__(
        self,
        title: str,
        *,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str,
        **kwargs,
    ):
        self.transform = transform
        self.__title = title
        self.__title_in_source = title if title_in_source is None else title_in_source
        self.__class_names = []
        self._cache_dir: Path = Path(cache_dir).expanduser().resolve() / "datasets"

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
        cache_dir: str,
        **kwargs,
    ):
        super().__init__(title, title_in_source=title_in_source, transform=transform, cache_dir=cache_dir)
        # Separate HF datasets from other sources
        self._cache_dir /= "huggingface"
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

            # Standardize the dataset features
            if "labels" in self._dataset.features:
                self._dataset = self._dataset.rename_column("labels", "label")

            self.class_names = self._dataset.info.features["label"].names
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: {self.title_in_source}") from e

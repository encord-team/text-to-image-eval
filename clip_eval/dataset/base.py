from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset

from clip_eval.constants import CACHE_PATH


class Split(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    ALL = auto()


class Dataset(TorchDataset, ABC):
    def __init__(
        self,
        title: str,
        *,
        split: Split = Split.ALL,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.transform = transform
        self.__title = title
        self.__title_in_source = title if title_in_source is None else title_in_source
        self.__split = split
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
    def split(self) -> Split:
        return self.__split

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

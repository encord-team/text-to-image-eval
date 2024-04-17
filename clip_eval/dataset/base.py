from abc import ABC, abstractmethod
from enum import StrEnum, auto
from os.path import relpath
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset as TorchDataset

from clip_eval.constants import CACHE_PATH, SOURCES_PATH

DEFAULT_DATASET_TYPES_LOCATION = (
    Path(relpath(str(__file__), SOURCES_PATH.DATASET_INSTANCE_DEFINITIONS)).parent / "types" / "__init__.py"
)


class Split(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    ALL = auto()


class DatasetDefinitionSpec(BaseModel):
    dataset_type: str
    title: str
    module_path: Path = DEFAULT_DATASET_TYPES_LOCATION
    split: Split | None = None
    title_in_source: str | None = None
    cache_dir: Path | None = None

    # Allow additional dataset configuration fields
    model_config = ConfigDict(extra="allow")


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

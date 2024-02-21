from abc import ABC, abstractmethod

from datasets import Split, load_dataset
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset, ABC):
    def __init__(self, title: str, title_in_source: str | None = None, *, transform=None, **kwargs):
        self.transform = transform
        self.__title = title
        self.__title_in_source = title if title_in_source is None else title_in_source

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

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def _setup(self, **kwargs):
        pass


class HFDataset(Dataset):
    def __init__(self, title: str, title_in_source: str | None = None, *, transform=None, **kwargs):
        super().__init__(title, title_in_source, transform=transform)
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
            self._dataset = load_dataset(self.title_in_source, split=split, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face: {self.title_in_source}") from e

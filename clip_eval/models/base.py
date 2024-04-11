from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from clip_eval.common.numpy_types import ClassArray, EmbeddingArray
from clip_eval.constants import CACHE_PATH


class Model(ABC):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.__title = title
        self.__title_in_source = title_in_source or title
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._check_device(device)
        self.__device = torch.device(device)
        if cache_dir is None:
            cache_dir = CACHE_PATH
        self._cache_dir = Path(cache_dir).expanduser().resolve() / "models" / title

    @property
    def title(self) -> str:
        return self.__title

    @property
    def title_in_source(self) -> str:
        return self.__title_in_source

    @property
    def device(self) -> torch.device:
        return self.__device

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        ...

    @abstractmethod
    def get_collate_fn(self) -> Callable[[Any], Any]:
        ...

    @abstractmethod
    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        ...

    @staticmethod
    def _check_device(device: str):
        # Check if the input device exists and is available
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unrecognized device: {device}")
        if not getattr(torch, device).is_available():
            raise ValueError(f"Unavailable device: {device}")

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor as HF_ClipProcessor
from transformers import SiglipModel as HF_SiglipModel
from transformers import SiglipProcessor as HF_SiglipProcessor

from clip_eval.common.numpy_types import ClassArray, EmbeddingArray


class CLIPModel(ABC):
    def __init__(
        self,
        title: str,
        title_in_source: str | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.__title = title
        self.__title_in_source = title_in_source or title
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._check_device(device)
        self.__device = torch.device(device)

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
    def _setup(self, **kwargs):
        pass

    @abstractmethod
    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, ClassArray]:
        ...

    @abstractmethod
    def get_transform(self) -> Callable[[Any], Any]:
        ...

    @staticmethod
    def _check_device(device: str):
        # Check if the input device exists and is available
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unrecognized device: {device}")
        if not getattr(torch, device).is_available():
            raise ValueError(f"Unavailable device: {device}")


class closed_CLIPModel(CLIPModel):
    def __init__(self, title: str, title_in_source: str, device: str | None = None) -> None:
        super().__init__(title, title_in_source, device)
        self._setup()

    def define_process_fn(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def _setup(self):
        self.model = HF_ClipModel.from_pretrained(self.title_in_source).to(self.device)  # type: ignore
        load_result = HF_ClipProcessor.from_pretrained(self.title_in_source)
        self.processor = load_result[0] if isinstance(load_result, tuple) else load_result
        self.process_fn = self.define_process_fn()

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, ClassArray]:
        tmp_embeddings = []
        tmp_labels = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                tmp_labels.append(batch["labels"])
                features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                emb = (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                tmp_embeddings.append(emb.to("cpu"))
        image_embeddings: EmbeddingArray = np.concatenate(tmp_embeddings, 0)
        class_array = torch.concatenate(tmp_labels)
        labels = class_array.numpy()
        return image_embeddings, labels

    def get_transform(self) -> Callable[[Any], Any]:
        return self.process_fn


class open_CLIPModel(CLIPModel):
    def __init__(
        self,
        title: str,
        title_in_source: str | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title, title_in_source, device, **kwargs)
        self._setup()

    def _setup(self, **kwargs):
        raise NotImplementedError("open Clip not implemented")

    def build_embedding(self, dataloader: DataLoader):
        raise NotImplementedError("open Clip not implemented")

    def get_transform(self) -> Callable[[Any], Any]:
        raise NotImplementedError("open Clip not implemented")


class SiglipModel(CLIPModel):
    def __init__(self, title: str, title_in_source: str | None = None, device: str | None = None, **kwargs) -> None:
        super().__init__(title, title_in_source, device, **kwargs)
        self._setup()

    def _setup(self, **kwargs):
        self.model = HF_SiglipModel.from_pretrained(self.title_in_source)
        self.processor = HF_SiglipProcessor.from_pretrained(self.title_in_source)
        self.process_fn = self.define_process_fn()

    def define_process_fn(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, ClassArray]:
        tmp_embeddings = []
        tmp_labels = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                tmp_labels.append(batch["labels"])
                features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                emb = (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                tmp_embeddings.append(emb.to("cpu"))
        image_embeddings: EmbeddingArray = np.concatenate(tmp_embeddings, 0)
        class_array = torch.concatenate(tmp_labels)
        labels = class_array.numpy()
        return image_embeddings, labels

    def get_transform(self) -> Callable[[Any], Any]:
        return self.process_fn

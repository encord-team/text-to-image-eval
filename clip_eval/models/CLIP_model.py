from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor as HF_ClipProcessor
from transformers import SiglipModel as HF_SiglipModel
from transformers import SiglipProcessor as HF_SiglipProcessor

from clip_eval.common.numpy_types import ClassArray, EmbeddingArray
from clip_eval.constants import CACHE_PATH


class CLIPModel(ABC):
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
    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, ClassArray]:
        ...

    @staticmethod
    def _check_device(device: str):
        # Check if the input device exists and is available
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unrecognized device: {device}")
        if not getattr(torch, device).is_available():
            raise ValueError(f"Unavailable device: {device}")


class ClosedCLIPModel(CLIPModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir)
        self._setup(**kwargs)

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def get_collate_fn(self) -> Callable[[Any], Any]:
        def collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(example["label"])

            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return {"pixel_values": pixel_values, "labels": labels}

        return collate_fn

    def _setup(self, **kwargs) -> None:
        self.model = HF_ClipModel.from_pretrained(
            self.title_in_source,
            cache_dir=self._cache_dir,
        ).to(self.device)  # type: ignore
        load_result = HF_ClipProcessor.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)
        self.processor = load_result[0] if isinstance(load_result, tuple) else load_result

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


class OpenCLIPModel(CLIPModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str,
        pretrained: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.pretrained = pretrained
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir, **kwargs)
        self._setup(**kwargs)

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [self.processor(i).to(self.device).unsqueeze(0) for i in images]
            return batch

        return process_fn

    def get_collate_fn(self) -> Callable[[Any], Any]:
        def collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(example["label"])

            torch_images = torch.stack(images)
            labels = torch.tensor(labels)
            return {"image": torch_images, "labels": labels}

        return collate_fn

    def _setup(self, **kwargs) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.title_in_source,
            pretrained=self.pretrained,
            cache_dir=self._cache_dir.as_posix(),
            **kwargs,
        )
        self.model = model.to(self.device)
        self.processor = preprocess

    def build_embedding(self, dataloader: DataLoader):
        tmp_embeddings = []
        tmp_labels = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                tmp_labels.append(batch["labels"])
                features = torch.stack([self.model.encode_image(image) for image in batch["image"]])
                emb = (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                tmp_embeddings.append(emb.to("cpu"))
        image_embeddings: EmbeddingArray = np.concatenate(tmp_embeddings, 0)
        class_array = torch.concatenate(tmp_labels)
        labels = class_array.numpy()
        return image_embeddings, labels


class SiglipModel(CLIPModel):
    def __init__(
        self,
        title: str,
        device: str | None = None,
        *,
        title_in_source: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(title, device, title_in_source=title_in_source, cache_dir=cache_dir, **kwargs)
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        self.model = HF_SiglipModel.from_pretrained(self.title_in_source, cache_dir=self._cache_dir).to(self.device)
        self.processor = HF_SiglipProcessor.from_pretrained(self.title_in_source, cache_dir=self._cache_dir)

    def get_transform(self) -> Callable[[dict[str, Any]], dict[str, list[Any]]]:
        def process_fn(batch) -> dict[str, list[Any]]:
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def get_collate_fn(self) -> Callable[[Any], Any]:
        def collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(example["label"])

            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return {"pixel_values": pixel_values, "labels": labels}

        return collate_fn

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

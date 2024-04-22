from collections.abc import Callable
from typing import Any

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tti_eval.common import ClassArray, EmbeddingArray
from tti_eval.dataset import Dataset
from tti_eval.model import Model


class OpenCLIPModel(Model):
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
            batch["image"] = [self.processor(i) for i in images]
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
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name=self.title_in_source,
            pretrained=self.pretrained,
            cache_dir=self._cache_dir.as_posix(),
            device=self.device,
            **kwargs,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name=self.title_in_source)

    def build_embedding(self, dataloader: DataLoader) -> tuple[EmbeddingArray, EmbeddingArray, ClassArray]:
        all_image_embeddings = []
        all_labels = []
        with torch.inference_mode():
            _dataset: Dataset = dataloader.dataset
            text = self.tokenizer(_dataset.text_queries).to(self.device)
            class_embeddings = self.model.encode_text(text, normalize=True).numpy(force=True)
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                image_features = self.model.encode_image(batch["image"].to(self.device), normalize=True)
                all_image_embeddings.append(image_features)
                all_labels.append(batch["labels"])
        image_embeddings = torch.concatenate(all_image_embeddings).numpy(force=True)
        labels = torch.concatenate(all_labels).numpy(force=True).astype(np.int32)
        return image_embeddings, class_embeddings, labels

from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, model_validator
from torch.utils.data import DataLoader

from clip_eval.constants import NPZ_KEYS
from clip_eval.dataset import Dataset, dataset_provider
from clip_eval.models import CLIPModel, model_provider

from .numpy_types import ClassArray, EmbeddingArray


class Embeddings(BaseModel):
    images: EmbeddingArray
    classes: EmbeddingArray | None = None
    labels: ClassArray

    @model_validator(mode="after")
    def validate_shapes(self) -> "Embeddings":
        N_images = self.images.shape[0]
        N_labels = self.labels.shape[0]
        if N_images != N_labels:
            raise ValueError(f"Differing number of images: {N_images} and labels: {N_labels}")

        if self.classes is None:
            return self

        *_, d1 = self.images.shape
        *_, d2 = self.classes.shape

        if d1 != d2:
            raise ValueError(
                f"Image {self.images.shape} and classe {self.classes.shape} embeddings should have same dimensionality"
            )
        return self

    @staticmethod
    def from_file(path: Path) -> "Embeddings":
        if not path.suffix == ".npz":
            raise ValueError(f"Embedding files should be `.npz` files not {path.suffix}")

        loaded = np.load(path)
        if NPZ_KEYS.IMAGE_EMBEDDINGS not in loaded and NPZ_KEYS.LABELS not in loaded:
            raise ValueError(f"Require both {NPZ_KEYS.IMAGE_EMBEDDINGS}, {NPZ_KEYS.LABELS} should be present in {path}")

        image_embeddings: EmbeddingArray = loaded[NPZ_KEYS.IMAGE_EMBEDDINGS]
        labels: ClassArray = loaded[NPZ_KEYS.LABELS]
        label_embeddings: EmbeddingArray | None = (
            loaded[NPZ_KEYS.CLASS_EMBEDDINGS] if NPZ_KEYS.CLASS_EMBEDDINGS in loaded else None
        )
        return Embeddings(images=image_embeddings, labels=labels, classes=label_embeddings)

    def to_file(self, path: Path) -> Path:
        if not path.suffix == ".npz":
            raise ValueError(f"Embedding files should be `.npz` files not {path.suffix}")
        to_store: dict[str, np.ndarray] = {
            NPZ_KEYS.IMAGE_EMBEDDINGS: self.images,
            NPZ_KEYS.LABELS: self.labels,
        }

        if self.classes is not None:
            to_store[NPZ_KEYS.CLASS_EMBEDDINGS] = self.classes

        np.savez_compressed(
            path,
            **to_store,
        )
        return path

    @staticmethod
    def from_embedding_definition(model_name: str, dataset_name: str) -> "Embeddings":
        model = model_provider.get_model(model_name)
        dataset = dataset_provider.get_dataset(dataset_name)
        embeddings = Embeddings.build_embedding(model, dataset)
        return embeddings

    @staticmethod
    def build_embedding(model: CLIPModel, dataset: Dataset, batch_size: int = 50) -> "Embeddings":
        features = list(set(dataset._dataset[dataset.label]))
        feature_map = {feature: i for i, feature in enumerate(features)}

        def _collate_fn(examples) -> dict[str, torch.Tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(feature_map[example[dataset.label]])

            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return {"pixel_values": pixel_values, "labels": labels}

        dataset.set_transform(model.get_transform())
        dataloader = DataLoader(dataset, collate_fn=_collate_fn, batch_size=batch_size)

        image_embeddings, labels = model.build_embedding(dataloader)
        embeddings = Embeddings(images=image_embeddings, labels=labels)
        return embeddings

    class Config:
        arbitrary_types_allowed = True

from pydantic import BaseModel, model_validator

from common.numpy_types import ClassArray, EmbeddingArray
from constants import NPZ_KEYS
from pathlib import Path
import numpy as np


class Embeddings(BaseModel):
    images: EmbeddingArray
    classes: EmbeddingArray | None = None
    labels: ClassArray

    @model_validator(mode="after")
    def validate_shapes(self) -> "Embeddings":
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
        to_store: dict[str, EmbeddingArray] = {
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

    class Config:
        arbitrary_types_allowed = True
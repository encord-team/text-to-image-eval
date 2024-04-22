import logging
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from clip_eval.constants import NPZ_KEYS, PROJECT_PATHS
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import AfterValidator

from .numpy_types import ClassArray, EmbeddingArray
from .string_utils import safe_str

SafeName = Annotated[str, AfterValidator(safe_str)]
logger = logging.getLogger("multiclips")


class Split(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
    ALL = auto()


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
                f"Image {self.images.shape} and class {self.classes.shape} embeddings should have same dimensionality"
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

    class Config:
        arbitrary_types_allowed = True


class EmbeddingDefinition(BaseModel):
    model: SafeName
    dataset: SafeName

    def _get_embedding_path(self, split: Split, suffix: str) -> Path:
        return Path(self.dataset) / f"{self.model}_{split}{suffix}"

    def embedding_path(self, split: Split) -> Path:
        return PROJECT_PATHS.EMBEDDINGS / self._get_embedding_path(split, ".npz")

    def get_reduction_path(self, reduction_name: str, split: Split):
        return PROJECT_PATHS.REDUCTIONS / self._get_embedding_path(split, f".{reduction_name}.2d.npy")

    def load_embeddings(self, split: Split) -> Embeddings | None:
        """
        Load embeddings for embedding configuration or return None
        """
        try:
            return Embeddings.from_file(self.embedding_path(split))
        except ValueError:
            return None

    def save_embeddings(self, embeddings: Embeddings, split: Split, overwrite: bool = False) -> bool:
        """
        Save embeddings associated to the embedding definition.
        Args:
            embeddings: The embeddings to store
            split: The dataset split that corresponds to the embeddings
            overwrite: If false, won't overwrite and will return False

        Returns:
            True iff file stored successfully

        """
        if self.embedding_path(split).is_file() and not overwrite:
            logger.warning(
                f"Not saving embeddings to file `{self.embedding_path(split)}` as overwrite is False and file exists"
            )
            return False
        self.embedding_path(split).parent.mkdir(exist_ok=True, parents=True)
        embeddings.to_file(self.embedding_path(split))
        return True

    def __str__(self):
        return self.model + "_" + self.dataset

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EmbeddingDefinition) and self.model == other.model and self.dataset == other.dataset

    def __hash__(self):
        return hash((self.model, self.dataset))

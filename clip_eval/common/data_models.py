import logging
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

from clip_eval.constants import PROJECT_PATHS
from clip_eval.dataset import DatasetProvider, Split
from clip_eval.models import ModelProvider

from .base import Embeddings
from .string_utils import safe_str

SafeName = Annotated[str, AfterValidator(safe_str)]
logger = logging.getLogger("multiclips")


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

    def build_embeddings(self, split: Split) -> Embeddings:
        model = ModelProvider.prepare().get_model(self.model)
        dataset = DatasetProvider.prepare().get_dataset(self.dataset, split)
        return Embeddings.build_embedding(model, dataset)

    def __str__(self):
        return self.model + "_" + self.dataset

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EmbeddingDefinition) and self.model == other.model and self.dataset == other.dataset

    def __hash__(self):
        return hash((self.model, self.dataset))


if __name__ == "__main__":
    def_ = EmbeddingDefinition(
        model="weird_this  with  / stuff \\whatever",
        dataset="hello there dataset",
    )
    def_.embedding_path(Split.TRAIN).parent.mkdir(exist_ok=True, parents=True)

    images = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    classes = np.random.randn(10, 20).astype(np.float32)
    emb = Embeddings(images=images, labels=labels, classes=classes)
    # emb.to_file(def_.embedding_path(Split.TRAIN))
    def_.save_embeddings(emb, split=Split.TRAIN, overwrite=True)
    new_emb = def_.load_embeddings(Split.TRAIN)

    assert new_emb is not None
    assert np.allclose(new_emb.images, images)
    assert np.allclose(new_emb.labels, labels)

    from pydantic import ValidationError

    try:
        Embeddings(
            images=np.random.randn(100, 20).astype(np.float32),
            labels=np.random.randint(0, 10, size=(100,)),
            classes=np.random.randn(10, 30).astype(np.float32),
        )
        raise AssertionError()
    except ValidationError:
        pass

    def_ = EmbeddingDefinition(
        model="clip",
        dataset="LungCancer4Types",
    )
    embeddings = def_.build_embeddings(Split.VALIDATION)
    def_.save_embeddings(embeddings, split=Split.VALIDATION, overwrite=True)

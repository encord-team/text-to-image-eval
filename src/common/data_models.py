import logging
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

from src.common.base import Embeddings
from src.common.string_utils import safe_str
from src.constants import PROJECT_PATHS

SafeName = Annotated[str, AfterValidator(safe_str)]
logger = logging.getLogger("multiclips")


class EmbeddingDefinition(BaseModel):
    model: SafeName
    dataset: SafeName

    def _get_embedding_path(self, suffix: str) -> Path:
        return Path(self.dataset) / f"{self.model}{suffix}"

    @property
    def embedding_path(self) -> Path:
        return PROJECT_PATHS.EMBEDDINGS / self._get_embedding_path(".npz")

    def get_reduction_path(self, reduction_name: str):
        return PROJECT_PATHS.REDUCTIONS / self._get_embedding_path(f".{reduction_name}.2d.npy")

    def load_embeddings(self) -> Embeddings | None:
        """
        Load embeddings for embedding configuration or return None
        """
        try:
            return Embeddings.from_file(self.embedding_path)
        except ValueError:
            return None

    def save_embeddings(self, embeddings: Embeddings, overwrite: bool = False) -> bool:
        """
        Save embeddings associated to the embedding definition.
        Args:
            embeddings: The embeddings to store
            overwrite: If false, won't overwrite and will return False

        Returns:
            True iff file stored successfully

        """
        if self.embedding_path.is_file() and not overwrite:
            logger.warning(
                f"Not saving embedding file {self.embedding_path} as overwrite is False and file exists already"
            )
            return False
        self.embedding_path.parent.mkdir(exist_ok=True, parents=True)
        embeddings.to_file(self.embedding_path)
        return True

    def build_embeddings(self) -> Embeddings:
        return Embeddings.from_embedding_definition(self.model, self.dataset)


if __name__ == "__main__":
    def_ = EmbeddingDefinition(
        model="weird_this  with  / stuff \\whatever",
        dataset="hello there dataset",
    )
    def_.embedding_path.parent.mkdir(exist_ok=True, parents=True)

    images = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    classes = np.random.randn(10, 20).astype(np.float32)
    emb = Embeddings(images=images, labels=labels, classes=classes)
    # emb.to_file(def_.embedding_path)
    def_.save_embeddings(emb, overwrite=True)
    new_emb = def_.load_embeddings()

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
    embeddings = def_.build_embeddings()
    def_.save_embeddings(embeddings)

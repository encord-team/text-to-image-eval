import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from common.base import Embeddings
from common.string_utils import safe_str
from constants import PROJECT_PATHS
from dataset import HFDataset
from models import CLIPModel

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
        except ValueError as e:
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

    def build_embedding(self, overwrite=False) -> bool:
        if self.embedding_path.is_file() and not overwrite:
            return False
        try:
            self.model = self._load_model()
        except Exception:
            raise Exception
        try:
            self.dataset = self._load_dataset()
        except Exception:
            raise Exception

        embeddings: Embeddings = self.model.embed(self.dataset)
        return self.save_embeddings(embeddings)

    # Possible that the below methods should return objects rather than modifying state
    def _load_model(self) -> CLIPModel:
        try:
            return CLIPModel.load_model(model_name=self.model)
        except Exception as e:
            raise e

    def _load_dataset(self) -> HFDataset:
        try:
            return HFDataset(dataset_name=self.dataset)
        except Exception as e:
            raise e


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
        assert False
    except ValidationError:
        assert True

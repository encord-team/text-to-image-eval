import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from src.constants import NPZ_KEYS, PROJECT_PATHS
from src.datasets import HFDataset
from src.models import CLIPModel
from src.types.numpy_types import ClassArray, EmbeddingArray
from src.types.string_utils import safe_str

SafeName = Annotated[str, AfterValidator(safe_str)]
logger = logging.getLogger("multiclips")


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
            raise ValueError(
                f"Embedding files should be `.npz` files not {path.suffix}"
            )

        loaded = np.load(path)
        if NPZ_KEYS.IMAGE_EMBEDDINGS not in loaded and NPZ_KEYS.LABELS not in loaded:
            raise ValueError(
                f"Require both {NPZ_KEYS.IMAGE_EMBEDDINGS}, {NPZ_KEYS.LABELS} should be present in {path}"
            )

        image_embeddings: EmbeddingArray = loaded[NPZ_KEYS.IMAGE_EMBEDDINGS]
        labels: ClassArray = loaded[NPZ_KEYS.LABELS]
        label_embeddings: EmbeddingArray | None = (
            loaded[NPZ_KEYS.CLASS_EMBEDDINGS]
            if NPZ_KEYS.CLASS_EMBEDDINGS in loaded
            else None
        )
        return Embeddings(
            images=image_embeddings, labels=labels, classes=label_embeddings
        )

    def to_file(self, path: Path) -> Path:
        if not path.suffix == ".npz":
            raise ValueError(
                f"Embedding files should be `.npz` files not {path.suffix}"
            )
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


class EmbeddingDefinition(BaseModel):
    model: SafeName
    dataset: SafeName

    def _get_embedding_path(self, suffix: str) -> Path:
        return Path(self.dataset) / f"{self.model}{suffix}"

    @property
    def embedding_path(self) -> Path:
        return PROJECT_PATHS.EMBEDDINGS / self._get_embedding_path(".npz")

    def get_reduction_path(self, reduction_name: str):
        return PROJECT_PATHS.REDUCTIONS / self._get_embedding_path(
            f".{reduction_name}.2d.npy"
        )

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

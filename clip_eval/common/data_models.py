import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

from clip_eval.constants import PROJECT_PATHS
from clip_eval.dataset import Split

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

    def __str__(self):
        return self.model + "_" + self.dataset

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EmbeddingDefinition) and self.model == other.model and self.dataset == other.dataset

    def __hash__(self):
        return hash((self.model, self.dataset))

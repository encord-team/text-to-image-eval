from pathlib import Path

from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from src.constants import CACHE_PATH
from src.utils import safe_str

SafeName = Annotated[str, AfterValidator(safe_str)]


class EmbeddingDefinition(BaseModel):
    model_name: SafeName
    dataset_name: SafeName

    def _filename(self, suffix: str) -> str:
        return f"{self.model_name}__{self.dataset_name}{suffix}"

    @property
    def embedding_path(self) -> Path:
        return CACHE_PATH / self._filename(".npy")

    def get_reduction_path(self, reduction_name: str):
        return CACHE_PATH / self._filename(f".{reduction_name}.2d.npy")

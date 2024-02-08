from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingDefinition:
    model_name: str
    dataset_name: str

    @property
    def embedding_path(self) -> Path:
        return

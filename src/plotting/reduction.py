from abc import abstractmethod

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from types import EmbeddingArray, EmbeddingDefinition, ReductionArray


class Reducer:
    def __init__(self, title: str) -> None:
        self._title = title

    @property
    def title(self) -> str:
        return self._title

    @abstractmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        ...

    def get_reduction(
        self,
        embedding_def: EmbeddingDefinition,
        force_recompute: bool = False,
        **kwargs,
    ) -> ReductionArray:
        embedding_file = embedding_def.get_reduction_path(self.title)
        if embedding_file.is_file() and not force_recompute:
            reduction: ReductionArray = np.load(embedding_file)
            return reduction

        elif not embedding_def.embedding_path.is_file():
            raise ValueError(f"{embedding_def} does not have embeddings stored ({embedding_def.embedding_path})")

        embeddings: EmbeddingArray = np.load(embedding_def.embedding_path)
        return self.reduce(embeddings)


class UMAPReducer(Reducer):
    def __init__(self) -> None:
        super().__init__("umap")

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, umap_seed: int | None = None, **kwargs) -> ReductionArray:
        reducer = umap.UMAP()
        return reducer.fit_transform(embeddings)


class TSNEReducer(Reducer):
    def __init__(self) -> None:
        super().__init__("tsne")

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        reducer = TSNE()
        return reducer.fit_transform(embeddings)


class PCAReducer(Reducer):
    def __init__(self) -> None:
        super().__init__("pca")

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings)


if __name__ == "__main__":
    import numpy as np

    embeddings = np.random.randn(100, 20)

    for cls in [UMAPReducer, TSNEReducer, PCAReducer]:
        print(cls().reduce(embeddings).shape)

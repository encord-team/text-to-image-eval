from abc import abstractmethod
from typing import Literal

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tti_eval.common import EmbeddingArray, EmbeddingDefinition, ReductionArray, Split


class Reducer:
    @classmethod
    @abstractmethod
    def title(cls) -> str:
        raise NotImplementedError("This abstract method returns the title of the reducer implemented in this class.")

    @classmethod
    @abstractmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        raise NotImplementedError("This abstract method contains the implementation for reducing embeddings.")

    @classmethod
    def get_reduction(
        cls,
        embedding_def: EmbeddingDefinition,
        split: Split,
        force_recompute: bool = False,
        save: bool = True,
        **kwargs,
    ) -> ReductionArray:
        reduction_file = embedding_def.get_reduction_path(cls.title(), split=split)
        if reduction_file.is_file() and not force_recompute:
            reduction: ReductionArray = np.load(reduction_file)
            return reduction

        elif not embedding_def.embedding_path(split).is_file():
            raise ValueError(
                f"{repr(embedding_def)} does not have embeddings stored ({embedding_def.embedding_path(split)})"
            )

        image_embeddings: EmbeddingArray = np.load(embedding_def.embedding_path(split))["image_embeddings"]
        reduction = cls.reduce(image_embeddings)
        if save:
            reduction_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(reduction_file, reduction)
        return reduction


class UMAPReducer(Reducer):
    @classmethod
    def title(cls) -> str:
        return "umap"

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, umap_seed: int | None = None, **kwargs) -> ReductionArray:
        reducer = umap.UMAP()
        return reducer.fit_transform(embeddings)


class TSNEReducer(Reducer):
    @classmethod
    def title(cls) -> str:
        return "tsne"

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        reducer = TSNE()
        return reducer.fit_transform(embeddings)


class PCAReducer(Reducer):
    @classmethod
    def title(cls) -> str:
        return "pca"

    @classmethod
    def reduce(cls, embeddings: EmbeddingArray, **kwargs) -> ReductionArray:
        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings)


__REDUCTIONS = {
    UMAPReducer.title(): UMAPReducer,
    TSNEReducer.title(): TSNEReducer,
    PCAReducer.title(): PCAReducer,
}
REDUCTIONS = Literal["umap"] | Literal["tsne"] | Literal["pca"]


def reduction_from_string(name: str) -> UMAPReducer | TSNEReducer | PCAReducer:
    if name not in __REDUCTIONS:
        raise KeyError(f"{name} not in set {set(__REDUCTIONS.keys())}")
    return __REDUCTIONS[name]


if __name__ == "__main__":
    embeddings = np.random.randn(100, 20)

    for cls in [UMAPReducer, TSNEReducer, PCAReducer]:
        print(cls.reduce(embeddings).shape)

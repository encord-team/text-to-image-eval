import logging
from typing import Any

import numpy as np
from autofaiss import build_index

from tti_eval.common import Embeddings

from .base import EvaluationModel

logger = logging.getLogger("multiclips")


class I2IRetrievalEvaluator(EvaluationModel):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
        k: int = 100,
    ) -> None:
        """
        Image-to-Image retrieval evaluator.

        For each training embedding (e_i), this evaluator computes the percentage (accuracy) of its k nearest neighbors
        from the validation embeddings that share the same class. It returns the mean percentage of correct nearest
        neighbors across all training embeddings.

        :param train_embeddings: Training embeddings used for evaluation.
        :param validation_embeddings: Validation embeddings used for similarity search setup.
        :param num_classes: Number of classes. If not specified, it will be inferred from the training embeddings.
        :param k: Number of nearest neighbors.

        :raises ValueError: If the build of the faiss index for similarity search fails.
        """
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="I2IR")
        self.k = min(k, len(validation_embeddings.images))

        class_ids, counts = np.unique(self._val_embeddings.labels, return_counts=True)
        self._class_counts = np.zeros(self.num_classes, dtype=np.int32)
        self._class_counts[class_ids] = counts

        index, self.index_infos = build_index(self._val_embeddings.images, save_on_disk=False, verbose=logging.ERROR)
        if index is None:
            raise ValueError("Failed to build an index for knn search")
        self._index = index

        logger.info("knn classifier index_infos", extra=self.index_infos)

    def evaluate(self) -> float:
        _, nearest_indices = self._index.search(self._train_embeddings.images, self.k)  # type: ignore
        nearest_classes = self._val_embeddings.labels[nearest_indices]

        # To compute retrieval accuracy, we ensure that a maximum of Q elements per sample are retrieved,
        # where Q represents the size of the respective class in the validation embeddings
        top_nearest_per_class = np.where(self._class_counts < self.k, self._class_counts, self.k)
        top_nearest_per_sample = top_nearest_per_class[self._train_embeddings.labels]

        # Add a placeholder value for indices outside the retrieval scope
        nearest_classes[np.arange(self.k) >= top_nearest_per_sample[:, np.newaxis]] = -1

        # Count the number of neighbours that match the class of the sample and compute the mean accuracy
        matches_per_sample = np.sum(nearest_classes == np.array(self._train_embeddings.labels)[:, np.newaxis], axis=1)
        accuracies = np.divide(
            matches_per_sample,
            top_nearest_per_sample,
            out=np.zeros_like(matches_per_sample, dtype=np.float64),
            where=top_nearest_per_sample != 0,
        )
        return accuracies.mean().item()

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {"k": 100}


if __name__ == "__main__":
    np.random.seed(42)
    train_embeddings = Embeddings(
        images=np.random.randn(80, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(80,)),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(20, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(20,)),
    )
    mean_accuracy = I2IRetrievalEvaluator(
        train_embeddings,
        val_embeddings,
        num_classes=10,
    ).evaluate()
    print(mean_accuracy)

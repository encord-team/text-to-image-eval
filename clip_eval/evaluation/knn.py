import logging
from typing import Any

import numpy as np
from autofaiss import build_index

from clip_eval.common.data_models import Embeddings
from clip_eval.common.numpy_types import ClassArray, ProbabilityArray
from clip_eval.evaluation.base import ClassificationModel
from clip_eval.evaluation.utils import softmax

logger = logging.getLogger("multiclips")


class WeightedKNNClassifier(ClassificationModel):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
        k: int = 11,
    ) -> None:
        """
        Weighted KNN classifier based on the provided embeddings and labels.
        Output "probabilities" is a softmax over weighted votes of class.

        Given q, a sample to predict, this function identifies the k nearest neighbors (e_i)
        with corresponding classes (y_i) from `train_embeddings.images` and `train_embeddings.labels`, respectively,
        and assigns a class vote of 1/||e_i - q||_2^2 for class y_i.
        The class with the highest vote count will be chosen.

        :param train_embeddings: Embeddings and their labels used for setting up the search space.
        :param validation_embeddings: Embeddings and their labels used for evaluating the search space.
        :param num_classes: Number of classes. If not specified, it will be inferred from the train labels.
        :param k: Number of nearest neighbors.

        :raises ValueError: If the build of the faiss index for KNN fails.
        """
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="wKNN")
        self.k = k

        index, self.index_infos = build_index(
            train_embeddings.images,
            metric_type="l2",
            save_on_disk=False,
            verbose=logging.ERROR,
        )
        if index is None:
            raise ValueError("Failed to build an index for knn search")
        self._index = index

        logger.info("knn classifier index_infos", extra=self.index_infos)

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {"k": 11}

    def predict(self) -> tuple[ProbabilityArray, ClassArray]:
        dists, nearest_indices = self._index.search(self._val_embeddings.images, self.k)  # type: ignore
        nearest_classes = np.take(self._train_embeddings.labels, nearest_indices)

        # Calculate class votes from the distances (avoiding division by zero)
        max_value = np.finfo(np.float32).max
        scores = np.divide(1, np.square(dists), out=np.full_like(dists, max_value), where=dists != 0)
        # NOTE: if self.k and self.num_classes are both large, this might become a big one.
        # We can shape of a factor self.k if we count differently here.
        n = len(self._val_embeddings.images)
        weighted_count = np.zeros((n, self.num_classes, self.k), dtype=np.float32)
        weighted_count[
            np.tile(np.arange(n), self.k),  # [0, 0, .., 0_k, 1, 1, .., 1_k, ..]
            nearest_classes.reshape(-1),  # [class numbers]
            np.tile(np.arange(self.k), n),  # [0, 1, .., k-1, 0, 1, .., k-1, ..]
        ] = scores.reshape(-1)
        probabilities = softmax(weighted_count.sum(-1))
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    np.random.seed(42)

    train_embeddings = Embeddings(
        images=np.random.randn(100, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(100,)),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(2,)),
    )
    knn = WeightedKNNClassifier(
        train_embeddings,
        val_embeddings,
        num_classes=10,
    )
    probs, pred_classes = knn.predict()
    print(probs)
    print(pred_classes)

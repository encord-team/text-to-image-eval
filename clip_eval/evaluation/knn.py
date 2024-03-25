import logging
from typing import Any

import numpy as np
from autofaiss import build_index

from clip_eval.common.data_models import Embeddings
from clip_eval.common.numpy_types import ClassArray, ProbabilityArray
from clip_eval.evaluation.base import ClassificationModel

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
        Weighted KNN classifier based on embeddings and labels given to the constructor.
        Output "probabilities" is a softmax over weighted votes of class.

        Args:
            embeddings: The embeddings to do similarity search against.
            labels: The labels associated to the embeddings
            k: Number of neighbors to use for voting. k neighbors e_i with classes y_i from `embeddings` and `labels`,
                respectively, will be identified and their class vote will be 1/||e_i, q||_2^2 for class y_i where q is
                the sample to predict. The one class with the highest vote will be chosen.
            num_classes: If not specified will be inferred from the labels.

        Raises:
            ValueError: If the faiss index fails to build.
        """
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="wKNN")
        self.k = k

        index, self.index_infos = build_index(train_embeddings.images, save_on_disk=False, verbose=logging.ERROR)
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

        # NOTE: if self.k and self.num_classes are both large, this might become a big one.
        # We can shape of a factor self.k if we count differently here.
        n = len(self._val_embeddings.images)
        weighted_count = np.zeros((n, self.num_classes, self.k), dtype=np.float32)
        weighted_count[
            np.tile(np.arange(n), (self.k,)).reshape(-1),  # [0, 0, .., 0_k, 1, 1, .., 1_k, ..]
            nearest_classes.reshape(-1),  # [class numbers]
            np.tile(np.arange(self.k), (n,)),  # [0, 1, .., k-1, 0, 1, .., k-1, ..]
        ] = 1 / dists.reshape(-1)
        probabilities = self.softmax(weighted_count.sum(-1))
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
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

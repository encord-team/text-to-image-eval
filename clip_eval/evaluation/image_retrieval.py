import logging
from typing import Any

import numpy as np
from autofaiss import build_index

from clip_eval.common.data_models import Embeddings
from clip_eval.evaluation.base import EvaluationModel

logger = logging.getLogger("multiclips")


class ImageRetrievalEvaluator(EvaluationModel):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
        k: int = 100,
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
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="image_retrieval")
        self.k = min(k, len(train_embeddings.images))
        unique_labels, label_counts = np.unique(train_embeddings.labels, return_counts=True)
        self._class_counts = dict(zip(unique_labels, label_counts, strict=True))

        index, self.index_infos = build_index(train_embeddings.images, save_on_disk=False, verbose=logging.ERROR)
        if index is None:
            raise ValueError("Failed to build an index for knn search")
        self._index = index

        logger.info("knn classifier index_infos", extra=self.index_infos)

    def evaluate(self) -> float:
        _, nearest_indices = self._index.search(self._val_embeddings.images, self.k)  # type: ignore
        nearest_classes = np.take(self._train_embeddings.labels, nearest_indices)

        accuracies = []
        for index, row in enumerate(nearest_classes):
            row_label = self._val_embeddings.labels[index]
            # Handle underrepresented classes that may have less than `self.k` elements, and classes missing in `train`
            row_k_nearest = min(self.k, self._class_counts.get(row_label, 0))
            unique_labels, label_counts = np.unique(row[:row_k_nearest], return_counts=True)
            row_counts = dict(zip(unique_labels, label_counts, strict=True))
            row_acc = row_counts.get(row_label, 0) / row_k_nearest if row_k_nearest != 0 else 0.0
            accuracies.append(row_acc)
        return np.array(accuracies).mean().item()

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {"k": 100}


if __name__ == "__main__":
    train_embeddings = Embeddings(
        images=np.random.randn(80, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(80,)),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(2,)),
    )
    image_retrieval = ImageRetrievalEvaluator(
        train_embeddings,
        val_embeddings,
        num_classes=10,
    )
    mean_accuracy = image_retrieval.evaluate()
    print(mean_accuracy)

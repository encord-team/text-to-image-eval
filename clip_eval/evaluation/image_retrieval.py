import logging
from abc import ABC, abstractmethod

import numpy as np
from autofaiss import build_index

from clip_eval.common import ProbabilityArray
from clip_eval.common.data_models import Embeddings
from clip_eval.common.numpy_types import ClassArray, EmbeddingArray
from clip_eval.evaluation.base import ClassificationModel

logger = logging.getLogger("multiclips")


class EvaluationModel(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        ...


class ImageRetrievalEvaluator(ClassificationModel, EvaluationModel):
    def __init__(
        self,
        embeddings: EmbeddingArray,
        labels: ClassArray,
        class_embeddings: EmbeddingArray | None = None,
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
        super().__init__("image_retrieval")
        self._labels = labels
        self.k = min(k, len(embeddings))

        self.num_classes = num_classes or labels.max() + 1
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique_labels, label_counts, strict=True))

        embeddings = self.normalize(embeddings)
        index, self.index_infos = build_index(embeddings, save_on_disk=False, verbose=logging.ERROR)

        if index is None:
            raise ValueError("Failed to build an index for knn search")
        self._index = index

        logger.info("knn classifier index_infos", extra=self.index_infos)

    @property
    def dim(self) -> int:
        return self._index.d

    def evaluate(self, embeddings: Embeddings) -> float:
        super()._check_dims(embeddings)
        img_embeddings = self.normalize(embeddings.images)

        _, nearest_indices = self._index.search(img_embeddings, self.k)  # type: ignore
        nearest_classes = np.take(self._labels, nearest_indices)

        all_acc = []
        for index, row in enumerate(nearest_classes):
            row_label = embeddings.labels[index]
            # Handle underrepresented classes that may have less than `self.k` elements, and classes missing in `train`
            row_k_nearest = min(self.k, self.class_counts.get(row_label, 0))
            unique_labels, label_counts = np.unique(row[:row_k_nearest], return_counts=True)
            row_counts = dict(zip(unique_labels, label_counts, strict=True))
            row_acc = row_counts.get(row_label, 0) / row_k_nearest if row_k_nearest != 0 else 0.0
            all_acc.append(row_acc)

        return np.array(all_acc).mean().item()

    def predict(self, embeddings: Embeddings) -> tuple[ProbabilityArray, ClassArray]:
        pass


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
        train_embeddings.images,
        train_embeddings.labels,
        num_classes=10,
    )
    avg_acc = image_retrieval.evaluate(val_embeddings)
    print(avg_acc)

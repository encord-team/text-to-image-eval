import logging

import numpy as np
from autofaiss import build_index

from src.evaluation.base import ClassificationModel
from src.types import ClassArray, EmbeddingArray, Embeddings, ProbabilityArray

logger = logging.getLogger("multiclips")


class WeightedKNNClassifier(ClassificationModel):
    def __init__(
        self,
        embeddings: EmbeddingArray,
        labels: ClassArray,
        k: int = 11,
        num_classes: int | None = None,
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
        super().__init__("zero_shot")
        self._labels = labels
        self.k = k

        self.num_classes = num_classes or labels.max() + 1

        embeddings = self.normalize(embeddings)
        index, self.index_infos = build_index(embeddings, save_on_disk=False, verbose=logging.ERROR)

        if index is None:
            raise ValueError("Failed to build an index for knn search")
        self._index = index

        logger.info(f"knn classifier index_infos", extra=self.index_infos)

    @property
    def dim(self) -> int:
        return self._index.d

    def predict(self, embeddings: Embeddings) -> tuple[ProbabilityArray, ClassArray]:
        super()._check_dims(embeddings)
        n, *_ = embeddings.images.shape
        img_embeddings = self.normalize(embeddings.images)

        dists, nearest_indices = self._index.search(img_embeddings, self.k)  # type: ignore
        nearest_classes = np.take(self._labels, nearest_indices)

        # NOTE: if self.k and self.num_classes are both large, this might become a big one.
        # We can shape of a factor self.k if we count differently here.
        weighted_count = np.zeros((n, self.num_classes, self.k), dtype=np.float32)
        weighted_count[
            np.tile(np.arange(n), (self.k,)).reshape(-1),  # [0, 0, .., 0_k, 1, 1, .., 1_k, ..]
            nearest_classes.reshape(-1),  # [class numbers]
            np.tile(np.arange(self.k), (n,)),  # [0, 1, .., k-1, 0, 1, .., k-1, ..]
        ] = 1 / dists.reshape(-1)
        probabilities = self.softmax(weighted_count.sum(-1))
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    knn = WeightedKNNClassifier(
        np.random.randn(100, 10).astype(np.float32),
        np.random.randint(0, 10, size=(100,)),
        num_classes=10,
    )
    embeddings = Embeddings(images=np.random.randn(2, 10).astype(np.float32))
    probs, cls = knn.predict(embeddings)
    print(probs)
    print(cls)

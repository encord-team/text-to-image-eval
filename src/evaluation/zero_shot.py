import numpy as np

from src.evaluation.base import ClassificationModel
from src.types import ClassArray, EmbeddingArray, Embeddings, ProbabilityArray


class ZeroShotClassifier(ClassificationModel):
    def __init__(self, class_embeddings: EmbeddingArray) -> None:
        super().__init__("zero_shot")
        self._class_embeddings: EmbeddingArray = self.normalize(class_embeddings)

    @property
    def dim(self) -> int:
        return self._class_embeddings.shape[-1]

    def predict(self, embeddings: Embeddings) -> tuple[ProbabilityArray, ClassArray]:
        super()._check_dims(embeddings)

        inner_products = self.normalize(embeddings.images) @ self._class_embeddings.T
        probabilities = self.softmax(inner_products)
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    zeroshot = ZeroShotClassifier(
        np.random.randn(10, 10).astype(np.float32),
    )
    embeddings = Embeddings(images=np.random.randn(2, 10).astype(np.float32))
    probs, cls = zeroshot.predict(embeddings)
    print(probs)
    print(cls)

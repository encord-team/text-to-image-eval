import numpy as np

from src.evaluation.base import ClassificationModel
from src.common import ClassArray, EmbeddingArray, Embeddings, ProbabilityArray


class ZeroShotClassifier(ClassificationModel):
    def __init__(
        self,
        embeddings: EmbeddingArray,
        labels: ClassArray,
        class_embeddings: EmbeddingArray,
        num_classes: int | None = None,
    ) -> None:
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
        np.random.randn(100, 10).astype(np.float32),
        np.random.randint(0, 10, size=(100,)),
        np.random.randn(20, 10).astype(np.float32),
        num_classes=20,
    )
    embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(2,)),
    )
    probs, cls = zeroshot.predict(embeddings)

    print(probs)
    print(cls)

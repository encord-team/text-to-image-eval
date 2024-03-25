import numpy as np

from clip_eval.common import ClassArray, Embeddings, ProbabilityArray
from clip_eval.evaluation.base import ClassificationModel


class ZeroShotClassifier(ClassificationModel):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
    ) -> None:
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="zero_shot")
        if self._train_embeddings.classes is None:
            raise ValueError("Expected class embeddings in `train_embeddings`, got `None`")

    @property
    def dim(self) -> int:
        return self._train_embeddings.classes.shape[-1]

    def predict(self) -> tuple[ProbabilityArray, ClassArray]:
        inner_products = self.normalize(self._val_embeddings.images) @ self._train_embeddings.classes.T
        probabilities = self.softmax(inner_products)
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    train_embeddings = Embeddings(
        images=np.random.randn(100, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(100,)),
        classes=np.random.randn(20, 10).astype(np.float32),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(2,)),
    )
    zeroshot = ZeroShotClassifier(
        train_embeddings,
        val_embeddings,
        num_classes=20,
    )
    probs, pred_classes = zeroshot.predict()
    print(probs)
    print(pred_classes)

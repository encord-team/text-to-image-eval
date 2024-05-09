import numpy as np

from tti_eval.common import ClassArray, Embeddings, ProbabilityArray
from tti_eval.evaluation.base import ClassificationModel
from tti_eval.evaluation.utils import softmax


class ZeroShotClassifier(ClassificationModel):
    @classmethod
    def title(cls) -> str:
        return "zero_shot"

    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
    ) -> None:
        """
        Zero-Shot classifier based on the provided embeddings and labels.

        :param train_embeddings: Embeddings and their labels used for setting up the search space.
        :param validation_embeddings: Embeddings and their labels used for evaluating the search space.
        :param num_classes: Number of classes. If not specified, it will be inferred from the train labels.
        """
        super().__init__(train_embeddings, validation_embeddings, num_classes)
        if self._train_embeddings.classes is None:
            raise ValueError("Expected class embeddings in `train_embeddings`, got `None`")

    @property
    def dim(self) -> int:
        return self._train_embeddings.classes.shape[-1]

    def predict(self) -> tuple[ProbabilityArray, ClassArray]:
        inner_products = self._val_embeddings.images @ self._train_embeddings.classes.T
        probabilities = softmax(inner_products)
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

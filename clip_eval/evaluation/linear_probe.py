import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from clip_eval.common import ClassArray, EmbeddingArray, Embeddings, ProbabilityArray
from clip_eval.evaluation.base import ClassificationModel

logger = logging.getLogger("multiclips")


class LinearProbeClassifier(ClassificationModel):
    def __init__(
        self,
        embeddings: EmbeddingArray,
        labels: ClassArray,
        class_embeddings: EmbeddingArray | None = None,
        num_classes: int | None = None,
        log_reg_params: dict[str, Any] | None = None,
        use_cross_validation: bool = False,
    ) -> None:
        """
        Logistic regression model based on embeddings and labels.

        Args:
            embeddings: The embeddings to do similarity search against.
            labels: The labels associated to the embeddings
            num_classes: If not specified will be inferred from the labels.
        """
        super().__init__("linear_probe")
        self.num_classes = num_classes or labels.max() + 1

        embeddings = self.normalize(embeddings)
        params = log_reg_params or {}
        self._dim = embeddings.shape[-1]
        self.classifier: LogisticRegressionCV | LogisticRegression
        if use_cross_validation:
            self.classifier = LogisticRegressionCV(
                Cs=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], random_state=0, **params
            ).fit(embeddings, labels)  # type: ignore
        else:
            self.classifier = LogisticRegression(random_state=0, **params).fit(embeddings, labels)

    @property
    def dim(self) -> int:
        return self._dim

    def predict(self, embeddings: Embeddings) -> tuple[ProbabilityArray, ClassArray]:
        super()._check_dims(embeddings)
        img_embeddings = self.normalize(embeddings.images)
        probabilities: ProbabilityArray = self.classifier.predict_proba(img_embeddings)  # type: ignore
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    model = LinearProbeClassifier(
        np.random.randn(100, 10).astype(np.float32),
        np.random.randint(0, 10, size=(100,)),
        num_classes=20,
    )
    embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 20, size=(2,)).astype(np.float32),
    )
    probs, cls = model.predict(embeddings)
    print(probs)
    print(cls)

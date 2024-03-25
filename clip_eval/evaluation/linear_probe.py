import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from clip_eval.common import ClassArray, Embeddings, ProbabilityArray
from clip_eval.evaluation.base import ClassificationModel

logger = logging.getLogger("multiclips")


class LinearProbeClassifier(ClassificationModel):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
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
        super().__init__(train_embeddings, validation_embeddings, num_classes, title="linear_probe")

        params = log_reg_params or {}
        self.classifier: LogisticRegressionCV | LogisticRegression
        if use_cross_validation:
            self.classifier = LogisticRegressionCV(
                Cs=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], random_state=0, **params
            ).fit(self._train_embeddings.images, self._train_embeddings.labels)  # type: ignore
        else:
            self.classifier = LogisticRegression(random_state=0, **params).fit(
                self._train_embeddings.images, self._train_embeddings.labels
            )

    def predict(self) -> tuple[ProbabilityArray, ClassArray]:
        probabilities: ProbabilityArray = self.classifier.predict_proba(self._val_embeddings.images)  # type: ignore
        return probabilities, np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    train_embeddings = Embeddings(
        images=np.random.randn(100, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(100,)),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(2, 10).astype(np.float32),
        labels=np.random.randint(0, 20, size=(2,)).astype(np.float32),
    )
    linear_probe = LinearProbeClassifier(
        train_embeddings,
        val_embeddings,
        num_classes=20,
    )
    probs, pred_classes = linear_probe.predict()
    print(probs)
    print(pred_classes)

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from clip_eval.common import ClassArray, Embeddings, ProbabilityArray
from clip_eval.common.numpy_types import DType


class EvaluationModelTitleInterface:
    # Enforce the evaluation models' title while removing its explicit mention in the `EvaluationModel` init.
    # This way, when `EvaluationModel` is used as a type hint, there won't be a warning about unfilled title.
    def __init__(self, title: str, **kwargs) -> None:
        self._title = title

    @property
    def title(self) -> str:
        return self._title


class EvaluationModel(EvaluationModelTitleInterface, ABC):
    def __init__(
        self,
        train_embeddings: Embeddings,
        validation_embeddings: Embeddings,
        num_classes: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Preprocessing the embeddings
        train_embeddings.images = self.normalize(train_embeddings.images)
        validation_embeddings.images = self.normalize(validation_embeddings.images)
        if train_embeddings.classes is not None:
            train_embeddings.classes = self.normalize(train_embeddings.classes)
        if validation_embeddings.classes is not None:
            validation_embeddings.classes = self.normalize(validation_embeddings.classes)

        self._train_embeddings = train_embeddings
        self._val_embeddings = validation_embeddings
        self._check_dims()
        self._num_classes = num_classes or train_embeddings.labels.max() + 1

    def _check_dims(self):
        """
        Raises error if dimensions doesn't match.
        """
        train_images_s, train_labels_s = self._train_embeddings.images.shape, self._train_embeddings.labels.shape
        if train_labels_s[0] != train_images_s[0]:
            raise ValueError(
                f"Expected `{train_images_s[0]}` samples in train's label embeddings, got `{train_labels_s[0]}`"
            )
        if self._train_embeddings.classes is not None and self._train_embeddings.classes.shape[-1] != self.dim:
            raise ValueError(
                f"Expected train's class embeddings with dimensions `{self.dim}`, "
                f"got `{self._train_embeddings.classes.shape[-1]}`"
            )

        val_images_s, val_labels_s = self._val_embeddings.images.shape, self._val_embeddings.labels.shape
        if val_labels_s[0] != val_images_s[0]:
            raise ValueError(
                f"Expected `{val_images_s[0]}` samples in validation's label embeddings, got `{val_labels_s[0]}`"
            )
        if val_images_s[-1] != self.dim:
            raise ValueError(
                f"Expected validation's image embeddings with dimensions `{self.dim}`, "
                f"got `{self._val_embeddings.images.shape[-1]}`"
            )
        if self._val_embeddings.classes is not None and self._val_embeddings.classes.shape[-1] != self.dim:
            raise ValueError(
                f"Expected validation's class embeddings with dimensions `{self.dim}`, "
                f"got `{self._val_embeddings.classes.shape[-1]}`"
            )

    @property
    def dim(self) -> int:
        """
        Expected embedding dimensionality.
        """
        return self._train_embeddings.images.shape[-1]

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {}

    @staticmethod
    def normalize(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def evaluate(self) -> float:
        ...


class ClassificationModel(EvaluationModel):
    @staticmethod
    def softmax(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
        z = x - x.max(axis=1, keepdims=True)
        numerator = np.exp(z)
        return np.exp(z) / np.sum(numerator, axis=1, keepdims=True)

    @abstractmethod
    def predict(self) -> tuple[ProbabilityArray, ClassArray]:
        ...

    def evaluate(self) -> float:
        _, y_hat = self.predict()
        acc: float = (y_hat == self._val_embeddings.labels).astype(float).mean().item()
        return acc

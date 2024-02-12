from abc import abstractmethod, abstractproperty
from typing import Any

import numpy as np
import numpy.typing as npt

from types import ClassArray, Embeddings, ProbabilityArray
from types.numpy_types import DType


class ClassificationModel:
    def __init__(self, title: str) -> None:
        self._title = title

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {}

    @property
    def title(self) -> str:
        return self._title

    def _check_dims(self, embeddings: Embeddings):
        """
        Raises error if dimensions doesn't match.
        """
        if embeddings.images.shape[-1] != self.dim:
            raise ValueError(
                f"Image embeddings ({embeddings.images.shape}) should have same dimensionality as the classifier expects ({self.dim})"
            )

    @abstractproperty
    def dim(self) -> int:
        """
        Expected dimensionality.
        """
        ...

    @staticmethod
    def softmax(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
        z = x - x.max(axis=1, keepdims=True)
        numerator = np.exp(z)
        return np.exp(z) / np.sum(numerator, axis=1, keepdims=True)

    @staticmethod
    def normalize(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    @abstractmethod
    def predict(self, embeddings: Embeddings) -> tuple[ProbabilityArray, ClassArray]:
        ...

import numpy as np
import numpy.typing as npt

from tti_eval.common.numpy_types import DType


def normalize(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def softmax(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
    z = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(z)
    return np.exp(z) / np.sum(numerator, axis=1, keepdims=True)

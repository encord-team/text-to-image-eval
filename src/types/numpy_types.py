from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

NDArray = Annotated[npt.NDArray[DType], Literal["N", "D"]]
EmbeddingArray = NDArray[np.float32]
"""
Numpy array of shape [N,D] embeddings.
"""
N2Array = Annotated[npt.NDArray[DType], Literal["N", 2]]
ReductionArray = N2Array[np.float32]
"""
Numpy array of shape [N,2] reductions of embeddings.
"""

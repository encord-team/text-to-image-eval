from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

EmbeddingArray = Annotated[npt.NDArray[DType], Literal["N", "D"]]
"""
Numpy array of shape [N,D] embeddings.
"""
ReductionArray = Annotated[npt.NDArray[DType], Literal["N", 2]]
"""
Numpy array of shape [N,2] reductions of embeddings.
"""

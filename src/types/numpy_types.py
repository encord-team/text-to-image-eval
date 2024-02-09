from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

NArray = Annotated[npt.NDArray[DType], Literal["N"]]
NCArray = Annotated[npt.NDArray[DType], Literal["N", "C"]]
NDArray = Annotated[npt.NDArray[DType], Literal["N", "D"]]
N2Array = Annotated[npt.NDArray[DType], Literal["N", 2]]

# Classes
ProbabilityArray = NCArray[np.float32]
ClassArray = NArray[np.int32]
"""
Numpy array of shape [N,C] probabilities where N is number of samples and C is number of classes.
"""

# Embeddings
EmbeddingArray = NDArray[np.float32]
"""
Numpy array of shape [N,D] embeddings where N is number of samples and D is the embedding size.
"""

# Reductions
ReductionArray = N2Array[np.float32]
"""
Numpy array of shape [N,2] reductions of embeddings.
"""

from multiprocessing.sharedctypes import RawArray
from typing import Sequence

import numpy as np


def create_shared_array(shape: Sequence[int], dtype: np.dtype) -> np.ndarray:
    ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    size = int(np.prod(shape))
    data = np.ctypeslib.as_array(RawArray(ctype, size))
    data.shape = shape
    return data.view(dtype)

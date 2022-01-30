from typing import Sequence, Union

import numpy as np

Observation = Union[np.ndarray, Sequence[np.ndarray]]
Action = Union[int, np.ndarray]

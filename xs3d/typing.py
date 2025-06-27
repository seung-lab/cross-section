from typing import Union

import numpy as np
import numpy.typing as npt

POINT_T = Union[tuple[int,int,int], tuple[int,int], npt.NDArray[np.integer]]
VECTOR_T = Union[tuple[float,float,float], tuple[float,float], npt.NDArray[np.float32]]

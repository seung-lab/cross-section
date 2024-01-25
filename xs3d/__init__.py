from typing import Tuple

from .twod import cross_sectional_area_2d
from .threed import cross_sectional_area_3d

import numpy as np

def cross_sectional_area(
  binimg:np.ndarray,
  pos:Tuple[int, int],
  vec:Tuple[float, float],
  anisotropy:Tuple[float, float] = [ 1.0, 1.0 ],
):
  if binimg.ndim == 2:
    return cross_sectional_area_2d(binimg, pos, vec, anisotropy)
  elif binimg.ndim == 3:
    return cross_sectional_area_3d(binimg, pos, vec, anisotropy)
  raise ValueError("dimensions not supported")
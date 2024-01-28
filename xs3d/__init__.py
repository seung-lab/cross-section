from typing import Sequence, Optional

from .twod import cross_sectional_area_2d
import fastxs3d

import numpy as np

def cross_sectional_area(
  binimg:np.ndarray,
  pos:Sequence[int],
  normal:Sequence[float],
  anisotropy:Optional[Sequence[float]] = None,
):
  if anisotropy is None:
    anisotropy = [ 1.0 ] * binimg.ndim

  pos = np.array(pos, dtype=np.float32)
  normal = np.array(normal, dtype=np.float32)
  anisotropy = np.array(anisotropy, dtype=np.float32)

  if binimg.dtype != bool:
    raise ValueError(f"A boolean image is required. Got: {binimg.dtype}")

  if np.any(anisotropy <= 0):
    raise ValueError(f"anisotropy values must be > 0. Got: {anisotropy}")

  if np.all(normal == 0):
    raise ValueError("normal vector must not be a null vector (all zeros).")

  binimg = np.asfortranarray(binimg)

  if binimg.ndim == 2:
    return cross_sectional_area_2d(binimg, pos, normal, anisotropy)
  elif binimg.ndim == 3:
    return fastxs3d.xsa(binimg, pos, normal, anisotropy)
  raise ValueError("dimensions not supported")
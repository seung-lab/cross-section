from typing import Sequence, Optional

from .twod import cross_sectional_area_2d
import fastxs3d

import numpy as np

def cross_sectional_area(
  binimg:np.ndarray,
  pos:Sequence[int],
  normal:Sequence[float],
  anisotropy:Optional[Sequence[float]] = None,
  return_contact:bool = False,
) -> float:
  """
  Find the cross sectional area for a given binary image, 
  point, and normal vector.

  binimg: a binary 2d or 3d numpy image (e.g. a bool datatype)
  pos: the point in the image from which to extract the section
    must be an integer (it's an index into the image).
    e.g. [5,10,2]
  normal: a vector normal to the section plane, does not
    need to be a unit vector. e.g. [sqrt(2)/2. sqrt(2)/2, 0]
  anisotropy: resolution of the x, y, and z axis
    e.g. [4,4,40] for an electron microscope image with 
    4nm XY resolution with a 40nm cutting plane in 
    serial sectioning.
  return_contact: if true, return a tuple of (area, contact)
    where area is the usual output and contact is non-zero if
    the section plane has contacted the edge of the image
    indicating the area may be an underestimate if you are
    working with a cutout of a larger image.

    Contact is an 8-bit bitfield that represents which image faces
    have been touched. The bits are organized as follows.

    0: 0 X     2: 0 Y     4: 0 Z      6: Unused
    1: Max X   3: Max Y   5: Max Z    7: Unused


  Returns: physical area covered by the section plane
  """
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
    area, contact = cross_sectional_area_2d(binimg, pos, normal, anisotropy)
  elif binimg.ndim == 3:
    area, contact = fastxs3d.xsa(binimg, pos, normal, anisotropy)
  else:
    raise ValueError("dimensions not supported")

  if return_contact:
    return (area, contact)
  else:
    return area


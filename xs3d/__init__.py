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
    area, contact = fastxs3d.area(binimg, pos, normal, anisotropy)
  else:
    raise ValueError("dimensions not supported")

  if return_contact:
    return (area, contact)
  else:
    return area

def cross_section(
  binimg:np.ndarray,
  pos:Sequence[int],
  normal:Sequence[float],
  anisotropy:Optional[Sequence[float]] = None,
  return_contact:bool = False,
) -> np.ndarray:
  """
  Compute which voxels are intercepted by a section plane
  (defined by a normal vector).

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

  Returns: float32 volume where each voxel's value is its
    contribution to the cross sectional area
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

  if binimg.ndim != 3:
    raise ValueError("dimensions not supported")
  
  section, contact = fastxs3d.section(binimg, pos, normal, anisotropy)

  if return_contact:
    return (section, contact)

  return section

def slice(
  labels:np.ndarray,
  pos:Sequence[int],
  normal:Sequence[float],
  anisotropy:Optional[Sequence[float]] = None,
  standardize_basis:bool = True,
) -> np.ndarray:
  """
  Compute which voxels are intercepted by a section plane
  and project them onto a plane.

  NB: The orientation of this projection is not guaranteed. 
  The axes can be reflected and transposed compared to what
  you might expect.

  labels: a binary 2d or 3d numpy image (e.g. a bool datatype)
  pos: the point in the image from which to extract the section
    must be an integer (it's an index into the image).
    e.g. [5,10,2]
  normal: a vector normal to the section plane, does not
    need to be a unit vector. e.g. [sqrt(2)/2. sqrt(2)/2, 0]
  anisotropy: resolution of the x, y, and z axis
    e.g. [4,4,40] for an electron microscope image with 
    4nm XY resolution with a 40nm cutting plane in 
    serial sectioning.
  standardize_basis: Tries (harder) to make the basis
    vectors closer to a standard basis and comport with
    human expectations (i.e. basis vectors point to the 
    right and up). However, this can cause discontinuities
    during smooth rotations.

    As of this writing, this feature reflects a basis vector
    if it is pointed > 90deg in opposition to the positive direction
    <1,1,1> or <-1,-1,-1> if the normal vector is pointed more in that
    direction.

  Returns: ndarray
  """
  if anisotropy is None:
    anisotropy = [ 1.0 ] * labels.ndim

  pos = np.array(pos, dtype=np.float32)
  normal = np.array(normal, dtype=np.float32)
  anisotropy = np.array(anisotropy, dtype=np.float32)

  if np.all(normal == 0):
    raise ValueError("normal vector must not be a null vector (all zeros).")

  labels = np.asfortranarray(labels)

  if labels.ndim != 3:
    raise ValueError(f"{labels.ndim} dimensions not supported")

  return fastxs3d.projection(labels, pos, normal, anisotropy, standardize_basis)






    



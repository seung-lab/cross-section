from typing import Iterator, Optional, Union

from .typing import POINT_T, VECTOR_T
from .twod import cross_sectional_area_2d
import fastxs3d

import numpy as np
import numpy.typing as npt


def cross_sectional_area(
  labels:npt.NDArray[np.number],
  pos:POINT_T,
  normal:VECTOR_T,
  anisotropy:Optional[VECTOR_T] = None,
  return_contact:bool = False,
  slow_method:bool = False,
  use_persistent_data:bool = False,
  segid:Optional[Union[int,float]] = None,
) -> Union[float, tuple[float, int]]:
  """
  Find the cross sectional area for a given binary image, 
  point, and normal vector.

  labels: a 2d or 3d numpy image
  segid: which label to consider foreground (default 1 for boolean images)
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

  slow_method: Calculate plane intersections at every
    voxel. Used for automated testing to ensure
    all locations are visited. Does not restrict analysis
    to a single connected component.

  use_persistent_data: Use a pre-allocated buffer for
    internally tacking visited positions. You allocate the
    buffer with xs3d.set_shape and clear it with xs3d.clear_shape.
    This can save about 20% of the time when repeatedly
    analyzing a shape.

  Returns: physical area covered by the section plane
  """
  if anisotropy is None:
    anisotropy = (1.0, 1.0, 1.0)
    if labels.ndim == 2:
      anisotropy = (1.0, 1.0)

  pos = np.asarray(pos, dtype=np.float32)
  normal = np.asarray(normal, dtype=np.float32)
  anisotropy = np.asarray(anisotropy, dtype=np.float32)

  if not np.issubdtype(labels.dtype, bool):
    if segid is None:
      raise ValueError(f"You must supply a segid for non-binary images.")
  else:
    segid = 1

  if np.any(anisotropy <= 0):
    raise ValueError(f"anisotropy values must be > 0. Got: {anisotropy}")

  if np.all(normal == 0):
    raise ValueError("normal vector must not be a null vector (all zeros).")

  labels = np.asfortranarray(labels)

  if labels.ndim == 2:
    area, contact = cross_sectional_area_2d(labels, pos, normal, anisotropy)
  elif labels.ndim == 3:
    area, contact = fastxs3d.area(
      labels, segid,
      pos, normal, anisotropy, 
      slow_method, use_persistent_data,
    )
  else:
    raise ValueError("dimensions not supported")

  if return_contact:
    return (area, contact)
  else:
    return area

def cross_section(
  binimg:npt.NDArray[np.bool_],
  pos:POINT_T,
  normal:VECTOR_T,
  anisotropy:Optional[VECTOR_T] = None,
  return_contact:bool = False,
  method:int = 0,
) -> Union[npt.NDArray[np.float32], tuple[npt.NDArray[np.float32], int]]:
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
    anisotropy = (1.0, 1.0, 1.0)
    if binimg.ndim == 2:
      anisotropy = (1.0, 1.0)

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
  
  section, contact = fastxs3d.section(
    binimg.view(np.uint8), 
    pos, normal, anisotropy, 
    method,
  )

  if return_contact:
    return (section, contact)

  return section

def slice(
  labels:npt.NDArray[np.integer],
  pos:POINT_T,
  normal:VECTOR_T,
  anisotropy:Optional[VECTOR_T] = None,
  standardize_basis:bool = True,
  crop:float = float('inf'),
) -> npt.NDArray[np.integer]:
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
  crop: distance in physical units to limit the slice to.
    This will reduce the size of the final output image to
    crop / min(anisotropy) in length.

    As of this writing, this feature reflects a basis vector
    if it is pointed > 90deg in opposition to the positive direction
    <1,1,1> or <-1,-1,-1> if the normal vector is pointed more in that
    direction.

  Returns: ndarray
  """
  assert crop >= 0.0

  if anisotropy is None:
    anisotropy = (1.0, 1.0, 1.0)
    if labels.ndim == 2:
      anisotropy = (1.0, 1.0)

  pos = np.asarray(pos, dtype=np.float32)
  normal = np.asarray(normal, dtype=np.float32)
  anisotropy = np.asarray(anisotropy, dtype=np.float32)

  if np.all(normal == 0):
    raise ValueError("normal vector must not be a null vector (all zeros).")

  if labels.ndim != 3:
    raise ValueError(f"{labels.ndim} dimensions not supported")

  if not labels.flags.f_contiguous:
    labels = np.ascontiguousarray(labels)

  return fastxs3d.projection(
    labels, pos, normal, 
    anisotropy, standardize_basis,
    crop
  )

def set_shape(image:npt.NDArray[np.integer]):
  """
  Allocate a buffer appropriately sized to this image for internal use.
  This can accelerate the area calculation if there are repeated queries
  against the same shape.

  It is very important that the persisted shape match the current input
  or memory issues can result.
  """
  fastxs3d.set_shape(image.shape[0], image.shape[1], image.shape[2])

def clear_shape():
  """Free the persisted memory from set_shape."""
  fastxs3d.clear_shape()




    



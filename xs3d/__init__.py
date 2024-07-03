from typing import Sequence, Optional, Tuple

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

def slice_path(
  labels:np.ndarray,
  path:Sequence[Sequence[int]],
  anisotropy:Optional[Sequence[float]] = None,
  smoothing:int = 1,
) -> np.ndarray:
  """
  Compute which voxels are intercepted by a section plane
  and project them onto a plane.

  NB: The orientation of this projection is not guaranteed. 
  The axes can be reflected and transposed compared to what
  you might expect.

  labels: a binary 2d or 3d numpy image (e.g. a bool datatype)
  path: a sequence of points in the image from which to extract the section
    must be an integer (it's an index into the image).
    e.g. [5,10,2]
  anisotropy: resolution of the x, y, and z axis
    e.g. [4,4,40] for an electron microscope image with 
    4nm XY resolution with a 40nm cutting plane in 
    serial sectioning.

  smoothing: number of verticies in the path to smooth the tangent
    vectors with.

  Returns: ndarray
  """
  if anisotropy is None:
    anisotropy = [ 1.0 ] * labels.ndim

  path = np.array(path, dtype=np.float32)

  if path.ndim != 2:
    raise ValueError("pos must be a sequence of x,y,z points.")

  if labels.ndim != 3:
    raise ValueError(f"{labels.ndim} dimensions not supported")

  anisotropy = np.array(anisotropy, dtype=np.float32)
  labels = np.asfortranarray(labels)

  # vectors aligned with the path
  tangents = (path[1:] - path[:-1]).astype(np.float32)
  tangents = np.concatenate([ tangents, [tangents[-1]] ])

  # Running the filter in the forward and then backwards
  # direction eliminates phase shift.
  tangents = _moving_average(tangents, smoothing)
  tangents = _moving_average(tangents[::-1], smoothing)[::-1]

  basis1s = (tangents[1:] - tangents[:-1]).astype(np.float32)
  basis1s = np.concatenate([ basis1s, [basis1s[-1]] ])

  basis2s = []

  basis1 = basis1s[0]
  if np.all(basis1 == 0):
    basis1 = np.cross(tangents[0], [1,0,0])
    if np.all(basis1 == 0):
      basis1 = np.cross(tangents[0], [0,1,0])

  basis1s[0] = basis1

  for i in range(1, len(basis1s)):
    if np.all(basis1s[i] == 0):
      basis1s[i] = basis1s[i-1]

  basis2 = np.cross(tangents[0], basis1)
  if np.all(basis2 == 0):
    basis2 = np.cross(tangents[0], [1,0,0])
    if np.all(basis2 == 0):
      basis2 = np.cross(tangents[0], [0,1,0])

  basis2s.append(basis2)

  for tangent, delta in zip(tangents[1:], basis1s[1:]):
    basis1 = delta
    basis2 = np.cross(tangent, basis1)
    if np.all(basis2 == 0):
      basis2 = basis2s[-1]
    basis2s.append(basis2)

  for i in range(len(basis1s)):
    basis1s[i] /= np.linalg.norm(basis1s[i])
    basis2s[i] /= np.linalg.norm(basis2s[i])

  slices = []
  from tqdm import tqdm
  for pos, basis1, basis2 in tqdm(zip(path, basis1s, basis2s)):
    slices.append(
      fastxs3d.projection_with_frame(
        labels, pos, 
        basis1, basis2, 
        anisotropy
      )
    )
  return slices

# From SO: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def _moving_average(a:np.ndarray, n:int, mode:str = "symmetric") -> np.ndarray:
  if n <= 0:
    raise ValueError(f"Window size ({n}), must be >= 1.")
  elif n == 1:
    return a

  if len(a) == 0:
    return a

  if a.ndim == 2:
    a = np.pad(a, [[n, n],[0,0]], mode=mode)
  else:
    a = np.pad(a, [n, n], mode=mode)

  ret = np.cumsum(a, dtype=float, axis=0)
  ret = (ret[n:] - ret[:-n])[:-n]
  ret /= float(n)
  return ret

def slice_path2(
  labels:np.ndarray,
  path:Sequence[Sequence[int]],
  anisotropy:Optional[Sequence[float]] = None,
  smoothing:int = 1,
  threshold:float = 1e-3,
) -> np.ndarray:
  """
  Compute which voxels are intercepted by a section plane
  that perpendicular to the path and project them onto a plane.

  NB: The orientation of this projection is not guaranteed. 
  The axes can be reflected and transposed compared to what
  you might expect.

  labels: a binary 2d or 3d numpy image (e.g. a bool datatype)
  path: a sequence of points in the image from which to extract the section
    must be an integer (it's an index into the image).
    e.g. [5,10,2]
  anisotropy: resolution of the x, y, and z axis
    e.g. [4,4,40] for an electron microscope image with 
    4nm XY resolution with a 40nm cutting plane in 
    serial sectioning.

  smoothing: number of verticies in the path to smooth the tangent
    vectors with.

  Returns: ndarray
  """
  if anisotropy is None:
    anisotropy = [ 1.0 ] * labels.ndim

  path = np.array(path, dtype=np.float32)

  if path.ndim != 2:
    raise ValueError("pos must be a sequence of x,y,z points.")

  if labels.ndim != 3:
    raise ValueError(f"{labels.ndim} dimensions not supported")

  anisotropy = np.array(anisotropy, dtype=np.float32)
  labels = np.asfortranarray(labels)

  # vectors aligned with the path
  tangents = (path[1:] - path[:-1]).astype(np.float32)
  tangents = np.concatenate([ tangents, [tangents[-1]] ])

  # Running the filter in the forward and then backwards
  # direction eliminates phase shift.
  tangents = _moving_average(tangents, smoothing)
  tangents = _moving_average(tangents[::-1], smoothing)[::-1]

  basis1s = np.cross(tangents[1:], tangents[:-1]) #.astype(np.float32)
  basis1s = np.concatenate([ basis1s, [basis1s[-1]] ])

  if np.all(abs(basis1s[0]) < threshold):
      for i in range(1, len(basis1s)):
          # If the current element does not have all values less than 10^-8
          if not np.all(abs(basis1s[i]) < threshold):
              basis1s[0] = basis1s[i]
              break

  for i in range(1, len(basis1s)):
    if np.all(abs(basis1s[0]) < threshold):
      basis1s[i] = basis1s[i-1]

  basis2s = np.cross(basis1s, tangents)

  for i in range(len(basis1s)):
    basis1s[i] /= np.linalg.norm(basis1s[i])
    basis2s[i] /= np.linalg.norm(basis2s[i])

  slices = []
  from tqdm import tqdm
  for pos, basis1, basis2 in tqdm(zip(path, basis1s, basis2s)):
    slices.append(
      fastxs3d.projection_with_frame(
        labels, pos, 
        basis1, basis2, 
        anisotropy
      )
    )
  return slices

def slice_with_frame(
  labels:np.ndarray,
  pos:Sequence[int],
  basis1:Sequence[float],
  basis2:Sequence[float],
  anisotropy:Optional[Sequence[float]] = None,
) -> np.ndarray:
  """
  Compute which voxels are intercepted by a section plane
  and project them onto a plane.

  NB: The orientation of this projection is not guaranteed. 
  The axes can be reflected and transposed compared to what
  you might expect.

  labels: a binary 2d or 3d numpy image (e.g. a bool datatype)

  anisotropy: resolution of the x, y, and z axis
    e.g. [4,4,40] for an electron microscope image with 
    4nm XY resolution with a 40nm cutting plane in 
    serial sectioning.

  Returns: ndarray
  """
  if anisotropy is None:
    anisotropy = [ 1.0 ] * labels.ndim

  pos = np.array(pos, dtype=np.float32)
  basis1 = np.array(basis1, dtype=np.float32)
  basis2 = np.array(basis2, dtype=np.float32)
  anisotropy = np.array(anisotropy, dtype=np.float32)

  labels = np.asfortranarray(labels)

  if labels.ndim != 3:
    raise ValueError(f"{labels.ndim} dimensions not supported")

  slice = fastxs3d.projection_with_frame(
        labels, pos, 
        basis1, basis2,
        anisotropy
      )
  return slice
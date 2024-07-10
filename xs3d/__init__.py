from typing import Sequence, Optional, Tuple, Generator

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
) -> Generator[np.ndarray, None, None]:
  """
  Compute which voxels are intercepted by a section plane
  and project them onto a plane.

  Why is it a generator? Because paths can be artibrarily long
  and this will avoid running out of memory (e.g. imagine a path
  that passes through every voxel in the image, for a 512x512x512
  uint64 volume, that could produce up to 550GB of image data).

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

  if len(path) == 1:
    yield slice(
      labels,
      pos=path[0],
      normal=[0,0,1],
      anisotropy=anisotropy,
    )
    return

  # vectors aligned with the path
  tangents = (path[1:] - path[:-1]).astype(np.float32)
  tangents = np.concatenate([ tangents, [tangents[-1]] ])

  # Running the filter in the forward and then backwards
  # direction eliminates phase shift.
  tangents = _moving_average(tangents, smoothing)
  tangents = _moving_average(tangents[::-1], smoothing)[::-1]

  basis1s = np.zeros([ len(tangents), 3 ], dtype=np.float32)
  basis2s = np.zeros([ len(tangents), 3 ], dtype=np.float32)

  if len(path) == 2:
    basis1s[0] = np.cross(tangents[0], [0,1,0])
    if np.isclose(np.linalg.norm(basis1s[0]), 0):
      basis1s[0] = np.cross(tangents[0], [1,0,0])
    basis2s[0] = np.cross(tangents[0], basis1s[0])

    basis1s[1] = basis1s[0]
    basis2s[1] = basis2s[0]
  else:
    basis1s[0] = np.cross(tangents[0], tangents[1])
    basis2s[0] = np.cross(basis1s[0] , tangents[0])

    for i in range(1, len(tangents)):
      R = _rotation_matrix_from_vectors(tangents[i-1], tangents[i])
      basis1s[i] = R @ basis1s[i-1].T
      basis2s[i] = R @ basis2s[i-1].T

  for i in range(len(basis1s)):
    basis1s[i] /= np.linalg.norm(basis1s[i])
    basis2s[i] /= np.linalg.norm(basis2s[i])

  for pos, basis1, basis2 in zip(path, basis1s, basis2s):
      yield fastxs3d.projection_with_frame(
        labels, pos, 
        basis1, basis2, 
        anisotropy
      )

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

def _rotation_matrix_from_vectors(vec1, vec2):
  """
  Find the rotation matrix that aligns vec1 to vec2
  :param vec1: A 3d "source" vector
  :param vec2: A 3d "destination" vector
  :return mat: A transformation matrix (3x3) which when applied to vec1, aligns it with vec2.

  Credit: help from chatgpt
  """
  a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)
  if s == 0:
    return np.eye(3)

  kmat = np.array([
    [0, -v[2], v[1]], 
    [v[2], 0, -v[0]], 
    [-v[1], v[0], 0]
  ])
  return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s * s))

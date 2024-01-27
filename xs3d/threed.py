from typing import List, Tuple

import cc3d
import numpy as np
import microviewer

def area_of_triangle(pt1, pt2, pt3) -> float:
  v1 = pt2 - pt1
  v2 = pt3 - pt1
  v3 = np.cross(v1,v2)
  return np.linalg.norm(v3) / 2.0

def area_of_quad(pt1, pt2, pt3, pt4) -> float:
  vecs = [ 
    pt2 - pt1,
    pt3 - pt1,
    pt4 - pt1,
  ]

  # remove the most distant point so we are
  # not creating a faulty quad based on the 
  # diagonal
  norms = [ np.linalg.norm(v) for v in vecs ]
  del vecs[np.argmax(norms)]
  v3 = np.cross(vecs[0],vecs[1])
  return np.linalg.norm(v3)

def nearest_point(x,y,z, pos, normal):
  # compute pos -> cur, subtract plane normal vec 
  # projection to get pos -> nearest point on plane
  cur = np.array([x,y,z], dtype=np.float32)
  cur_vec = cur - pos
  to_plane_vec = np.dot(cur_vec, normal) * normal
  cur_plane_vec = cur_vec - to_plane_vec
  return pos + cur_plane_vec

def get_ccl(
  binimg:np.ndarray, 
  pos:np.ndarray, 
  normal:np.ndarray,
) -> np.ndarray:

  ccl = np.zeros(binimg.shape, order="F", dtype=bool)

  sx, sy, sz = binimg.shape
  for z in range(sz):
    for y in range(sy):
      for x in range(sx):
        if binimg[x,y,z] == False:
          continue

        nearest_pt = nearest_point(x,y,z, pos, normal)
        
        v0 = float(x) - 0.5
        v1 = float(x) + 0.5

        if nearest_pt[0] < v0 or nearest_pt[0] > v1:
          continue

        h0 = float(y) - 0.5
        h1 = float(y) + 0.5
        
        if nearest_pt[1] < h0 or nearest_pt[1] > h1:
          continue

        d0 = float(z) - 0.5
        d1 = float(z) + 0.5
        
        if nearest_pt[2] < d0 or nearest_pt[2] > d1:
          continue

        tocoord = lambda p_i, si: min(int(round(p_i)), si - 1)

        px = tocoord(nearest_pt[0], sx) 
        py = tocoord(nearest_pt[1], sy)
        pz = tocoord(nearest_pt[2], sz)

        ccl[px, py, pz] = binimg[px,py,pz]

  return cc3d.connected_components(ccl, connectivity=26)

def check_intersections(
  x:int, y:int, z:int,
  pos:np.ndarray,
  normal:np.ndarray,
) -> List[List[float]]:

  # corners
  c = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
  ], dtype=np.float32)

  pipes = np.array([
    (c[4] - c[0]),
    (c[2] - c[0]), 
    (c[6] - c[2]),
    (c[6] - c[4]),

    (c[7] - c[3]),
    (c[3] - c[1]),
    (c[5] - c[1]),
    (c[7] - c[5]),

    (c[3] - c[2]),
    (c[1] - c[0]),
    (c[7] - c[6]),
    (c[5] - c[4]),
  ], dtype=np.float32)

  pipe_points = np.array([
    c[0], c[0], c[2], c[4],
    c[3], c[3], c[1], c[5],
    c[2], c[0], c[6], c[4],
  ], dtype=np.float32)

  pipe_points += np.array([x,y,z], dtype=np.float32) - 0.5

  pts = []
  for cur, vec in zip(pipe_points, pipes):
    cur_vec = cur - pos
    proj = np.dot(cur_vec, normal)
    if proj == 0: # corner is on the plane
      pts.append(cur)
      continue

    proj2 = np.dot(vec, normal)
    # if traveling parallel to plane but
    # not on the plane
    if proj2 == 0:
      continue

    t = proj2 / proj
    nearest_pt = cur + t * vec

    if nearest_pt[0] > x+0.5 or nearest_pt[0] < x-0.5:
      continue
    elif nearest_pt[1] > y+0.5 or nearest_pt[1] < y-0.5:
      continue
    elif nearest_pt[2] > z+0.5 or nearest_pt[2] < z-0.5:
      continue

    pts.append(nearest_pt)

  return np.unique(pts, axis=0)

def cross_sectional_area_3d(
  binimg:np.ndarray, 
  pos:Tuple[int, int, int], 
  vec:Tuple[float, float, float], 
  anisotropy:Tuple[float, float, float] = [ 1.0, 1.0, 1.0 ],
) -> float:

  sx, sy, sz = binimg.shape

  if pos[0] >= sx or pos[0] < 0:
    raise ValueError(f"{pos[0]} is outside of the x-range 0 to {sx}.")

  if pos[1] >= sy or pos[1] < 0:
    raise ValueError(f"{pos[1]} is outside of the y-range 0 to {sy}.")

  if pos[2] >= sz or pos[2] < 0:
    raise ValueError(f"{pos[1]} is outside of the y-range 0 to {sy}.")

  if binimg[int(pos[0]), int(pos[1]), int(pos[2])] == False:
    return 0.0

  pos = np.array(pos, dtype=np.float32)
  normal = np.array(vec, dtype=np.float32) / np.linalg.norm(vec)

  ccl = get_ccl(binimg, pos, normal)
  label = ccl[int(pos[0]), int(pos[1]), int(pos[2])]

  total = 0.0
  for z in range(sz):
    for y in range(sy):
      for x in range(sx):
        if ccl[x,y,z] != label:
          continue

        pts = check_intersections(x,y,z, pos, normal)

        if len(pts) < 3:
          continue
        elif len(pts) > 4:
          raise ValueError(x,y,z,pts)

        if len(pts) == 3:
          total += area_of_triangle(*pts)
        else:
          total += area_of_quad(*pts)

  return total







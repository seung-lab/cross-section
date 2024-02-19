from typing import List, Tuple

import cc3d
import numpy as np

def nearest_point(
  pt:Tuple[int,int], 
  m:float, 
  b:float,
) -> List[float]:
  # get eqn of normal line
  m_n = np.inf
  if m != 0:
    m_n = -1.0 / m
  b_n = pt[1] - m_n * pt[0]

  # compute the intersection
  p_x = (b - b_n) / (m_n - m)
  p_y = m * p_x + b

  return [p_x, p_y]

def get_ccl(
  binimg:np.ndarray, 
  pos:Tuple[int,int], 
  vec:Tuple[float,float],
) -> np.ndarray:
  slope = np.inf
  if vec[1] != 0:
    slope = -vec[0] / vec[1]
  b = pos[1] - slope * pos[0]

  ccl = np.zeros(binimg.shape, order="F", dtype=bool)

  sx, sy = binimg.shape
  for y in range(sy):
    for x in range(sx):
      if binimg[x,y] == False:
        continue

      if slope == 0:
        p_x = x
        p_y = pos[1]
      elif slope == np.inf:
        p_x = pos[0]
        p_y = y
      else:
        p_x, p_y = nearest_point([x,y], slope, b)

      v0 = float(x) - 0.5
      v1 = float(x) + 0.5

      if p_x < v0 or p_x > v1:
        continue

      h0 = float(y) - 0.5
      h1 = float(y) + 0.5
      
      if p_y < h0 or p_y > h1:
        continue

      tocoord = lambda p_i, si: min(int(round(p_i)), si - 1)

      px = tocoord(p_x, sx) 
      py = tocoord(p_y, sy)

      ccl[px, py] = binimg[px,py]

  return cc3d.connected_components(ccl, connectivity=8)

def cross_sectional_area_2d(
  binimg:np.ndarray, 
  pos:Tuple[int, int], 
  vec:Tuple[float, float], 
  anisotropy:Tuple[float, float] = [ 1.0, 1.0 ],
) -> float:

  sx, sy = binimg.shape

  if pos[0] >= sx or pos[0] < 0:
    return [0.0, 0b00111111]

  if pos[1] >= sy or pos[1] < 0:
    return [0.0, 0b00111111]

  if binimg[int(pos[0]), int(pos[1])] == False:
    return [0.0, 0b00111111]

  nhat = np.array([ -vec[1], vec[0] ], dtype=np.float32)
  nhat = nhat / np.sqrt(nhat[0] ** 2 + nhat[1] ** 2)

  ccl = get_ccl(binimg, pos, vec)
  label = ccl[int(pos[0]), int(pos[1])]

  total = 0.0

  if nhat[0] == 0:
    slope = np.inf
  else:
    slope = nhat[1] / nhat[0]

  wx, wy = float(anisotropy[0]), float(anisotropy[1])

  y_intercept = pos[1] - slope * pos[0]
  contact = 0

  for y in range(sy):
    for x in range(sx):
      if ccl[x,y] != label:
        continue

      contact |= (x == 0)
      contact |= (x == sx-1) << 1
      contact |= (y == 0) << 2
      contact |= (y == sy-1) << 3

      h0 = float(y) - 0.5
      h1 = float(y) + 0.5
      v0 = float(x) - 0.5
      v1 = float(x) + 0.5

      pts = []

      if slope == 0:
        if y == pos[1]:
          pts.append([v0, y])
          pts.append([v1, y])
      elif slope in (np.inf, -np.inf):
        if x == pos[0]:
          pts.append([x, h0])
          pts.append([x, h1])
      else:
        x0 = (h0 - y_intercept) / slope
        x1 = (h1 - y_intercept) / slope
        y0 = v0 * slope + y_intercept
        y1 = v1 * slope + y_intercept

        if y0 >= h0 and y0 <= h1:
          pts.append([v0, y0])
        if y1 >= h0 and y1 <= h1:
          pts.append([v1, y1])

        if x0 >= v0 and x0 <= v1:
          pts.append([x0, h0])
        if x1 >= v0 and x1 <= v1:
          pts.append([x1, h1])

      pts = np.unique(pts, axis=0)

      if len(pts) < 2:
        continue
      elif len(pts) > 2:
        raise ValueError(pts)

      pt1, pt2 = pts

      px = wx * (pt1[0] - pt2[0])
      py = wy * (pt1[1] - pt2[1])

      total += np.sqrt(px*px + py*py)

  return total, contact



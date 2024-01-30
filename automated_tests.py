import numpy as np
import xs3d


def test_single_voxel():
	voxel = np.ones([1,1,1], dtype=bool, order="F")

	area = xs3d.cross_sectional_area(voxel, [0,0,0], [0,0,1])
	assert area == 1
	area = xs3d.cross_sectional_area(voxel, [0,0,0], [0,1,0])
	assert area == 1
	area = xs3d.cross_sectional_area(voxel, [0,0,0], [1,0,0])
	assert area == 1

	area = xs3d.cross_sectional_area(voxel, [0,0,0], [1,1,0])
	assert np.isclose(area, np.sqrt(2))

	area = xs3d.cross_sectional_area(voxel, [0,0,0], [0,1,1])
	assert np.isclose(area, np.sqrt(2))

	area = xs3d.cross_sectional_area(voxel, [0,0,0], [1,0,1])
	assert np.isclose(area, np.sqrt(2))

	area = xs3d.cross_sectional_area(voxel, [0,0,0], [1,1,1])
	tri = np.sqrt(3) / 2 * ((0.5) ** 2)
	hexagon = 6 * tri
	assert np.isclose(area, hexagon)

	# outside the voxel
	area = xs3d.cross_sectional_area(voxel, [1,0,0], [1,0,0])
	assert area == 0
	area = xs3d.cross_sectional_area(voxel, [-1,0,0], [1,0,0])
	assert area == 0

	# arbitrary angles
	for incr in range(11):
		area = xs3d.cross_sectional_area(voxel, [0,0,0], [incr * 0.1,0,1])
		assert area >= 1
		assert area <= np.sqrt(2)


def test_ccl():
	img = np.zeros([10,10,10], dtype=bool, order="F")

	img[:3,:3,:3] = True
	img[6:,6:,:3] = True

	area = xs3d.cross_sectional_area(img, [1,1,1], [0,0,1])
	assert area == 9

	area = xs3d.cross_sectional_area(img, [7,7,1], [0,0,1])
	assert area == 16

	area = xs3d.cross_sectional_area(img, [7,7,5], [0,0,1])
	assert area == 0


def test_sphere():
	d = 100
	r = d/2
	img = np.zeros([125,125,125], dtype=bool, order="F")
	offset = 63

	def dist(x,y,z):
		nonlocal r
		x = x - offset
		y = y - offset
		z = z - offset
		return np.sqrt(x*x + y*y + z*z)

	for z in range(img.shape[2]):
		for y in range(img.shape[1]):
			for x in range(img.shape[0]):
				if dist(x,y,z) <= r:
					img[x,y,z] = True

	def angle(theta):
		return [ np.cos(theta), np.sin(theta), 0 ]

	pos = (offset, offset, offset)
	smoothness = ((r-1)**2) / (r**2)

	prev_area = xs3d.cross_sectional_area(img, pos, [1,0,0])

	for theta in range(0,50):
		normal = angle(theta / 50 * 2 * np.pi)
		area = xs3d.cross_sectional_area(img, pos, normal)

		assert area > np.pi * (r-1.5) * (r-1.5)
		assert area <= np.pi * (r+0.5) * (r+0.5)
		ratio = abs(area - prev_area) / area
		assert ratio < smoothness

		prev_area = area


	def angle2(theta):
		return [ 0, np.cos(theta), np.sin(theta) ]

	pos = (offset, offset, offset)

	prev_area = xs3d.cross_sectional_area(img, pos, [1,0,0])

	for theta in range(0,50):
		normal = angle2(theta / 50 * 2 * np.pi)
		area = xs3d.cross_sectional_area(img, pos, normal)

		assert area > np.pi * (r-1.5) * (r-1.5)
		assert area <= np.pi * (r+0.5) * (r+0.5)
		ratio = abs(area - prev_area) / area
		assert ratio < smoothness

		prev_area = area


















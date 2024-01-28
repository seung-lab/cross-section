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








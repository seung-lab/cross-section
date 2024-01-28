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




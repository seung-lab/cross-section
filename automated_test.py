import pytest

import numpy as np
import xs3d


@pytest.mark.parametrize("anisotropy", [
    [1,1,1],
    [2,2,2],
    [1000,1000,1000],
    [0.0001,0.0001,0.0001],
    [1, 1, 0.001],
    [1, 0.001, 1],
    [0.001, 1, 1],
])
def test_single_voxel(anisotropy):
    voxel = np.ones([1,1,1], dtype=bool, order="F")

    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [0,0,1], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, anisotropy[0] * anisotropy[1])
    assert contact > 0
    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [0,1,0], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, anisotropy[0] * anisotropy[2])
    assert contact > 0
    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [1,0,0], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, anisotropy[1] * anisotropy[2])
    assert contact > 0

    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [1,1,0], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, 
        (
            np.sqrt(anisotropy[0] * anisotropy[0] + anisotropy[1] * anisotropy[1]) 
            * anisotropy[2]
        )
    )
    assert contact > 0

    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [0,1,1], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, (
        np.sqrt(anisotropy[1] * anisotropy[1] + anisotropy[2] * anisotropy[2]) 
        * anisotropy[0]
    ))
    assert contact > 0

    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [1,0,1], anisotropy,
        return_contact=True
    )
    assert np.isclose(area, (
        np.sqrt(anisotropy[0] * anisotropy[0] + anisotropy[2] * anisotropy[2]) 
        * anisotropy[1]
    ))
    assert contact > 0

    area, contact = xs3d.cross_sectional_area(
        voxel, [0,0,0], [1,1,1], anisotropy,
        return_contact=True
    )
    tri = lambda s: np.sqrt(3) / 8 * (s ** 2)

    if 0.001 in anisotropy:
        # collapses to a 2D shape
        hexagon = 0.75 # 1/4 + 1/4 + 1/8 + 1/8
    else:
        hexagon = 2 * sum([ tri(a) for a in anisotropy ])
    assert np.isclose(area, hexagon)
    assert contact > 0

    # outside the voxel
    area, contact = xs3d.cross_sectional_area(
        voxel, [1,0,0], [1,0,0],  anisotropy,
        return_contact=True
    )
    assert area == 0
    assert contact == False
    area = xs3d.cross_sectional_area(voxel, [-1,0,0], [1,0,0], anisotropy)
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

    area, contact = xs3d.cross_sectional_area(img, [1,1,1], [0,0,1], return_contact=True)
    assert area == 9
    assert contact > 0

    area = xs3d.cross_sectional_area(img, [7,7,1], [0,0,1])
    assert area == 16

    area = xs3d.cross_sectional_area(img, [7,7,5], [0,0,1])
    assert area == 0

    img[:4,:4,:4] = True
    img[1,1,1] = False

    area = xs3d.cross_sectional_area(img, [0,1,1], [0,0,1])
    assert area == 15

    img[2:4,2:4,6:8] = True
    area, contact = xs3d.cross_sectional_area(img, [2,2,6], [0,0,1], return_contact=True)
    assert area == 4
    assert contact == False

def test_8_connectivity():
    img = np.zeros([4,4,3], dtype=bool, order="F")
    img[0,0] = True
    img[1,1] = True
    img[2,2] = True
    img[3,3] = True

    img[3,3,1] = True

    area = xs3d.cross_sectional_area(img, [0,0,0], [0,0,1])
    assert area == 4

    img = np.zeros([4,4,3], dtype=bool, order="F")
    img[3,0] = True
    img[2,1] = True
    img[1,2] = True
    img[3,3] = True

    img[3,3,1] = True

    area = xs3d.cross_sectional_area(img, [3,0,0], [0,0,1])
    assert area == 3

    img = np.zeros([4,4,3], dtype=bool, order="F")
    img[3,0] = True
    img[1,1] = True
    img[0,2] = True
    img[3,3] = True

    img[3,3,1] = True

    area = xs3d.cross_sectional_area(img, [3,0,0], [0,0,1])
    assert area == 1



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

    for theta in range(0,720):
        normal = angle(theta / 720 * 2 * np.pi)
        area, contact = xs3d.cross_sectional_area(img, pos, normal, return_contact=True)

        assert area > np.pi * (r-0.5) * (r-0.5)
        assert area <= np.pi * (r+0.5) * (r+0.5)
        ratio = abs(area - prev_area) / area
        assert ratio < smoothness
        assert contact == False

        prev_area = area


    def angle2(theta):
        return [ 0, np.cos(theta), np.sin(theta) ]

    pos = (offset, offset, offset)

    prev_area = xs3d.cross_sectional_area(img, pos, [1,0,0])

    for theta in range(0,720):
        normal = angle2(theta / 720 * 2 * np.pi)
        area = xs3d.cross_sectional_area(img, pos, normal)

        assert area > np.pi * (r-0.5) * (r-0.5)
        assert area <= np.pi * (r+0.5) * (r+0.5)
        ratio = abs(area - prev_area) / area
        assert ratio < smoothness

        prev_area = area

    def angle3(theta, phi):
        return [ phi, np.cos(theta), np.sin(theta) ]

    pos = (offset, offset, offset)

    prev_area = xs3d.cross_sectional_area(img, pos, [1,0,0])

    for theta in range(0, 1000):
        for phi in range(0, 500):
            normal = angle3(theta/1000, phi/500)
            area = xs3d.cross_sectional_area(img, pos, normal)

            assert area > np.pi * (r-0.5) * (r-0.5)
            assert area <= np.pi * (r+0.5) * (r+0.5)
            ratio = abs(area - prev_area) / area
            assert ratio < smoothness

            prev_area = area

def test_off_angle():
    binimg = np.ones([2,2,2], dtype=bool)
    pos = [1,1,1]
    normal = [ 0.92847669, -0.37139068, 0]

    approximate_area = 4 * np.sqrt(1 + (normal[1]/normal[0]) ** 2)

    area = xs3d.cross_sectional_area(binimg, pos, normal)
    assert abs(area - approximate_area) < 0.001


def test_5x5():
    binimg = np.ones([5,5,1], dtype=bool)

    area = xs3d.cross_sectional_area(binimg, [0,0,0], [0,0,1])

    assert area == 25


def test_symmetric_normals():
    labels = np.ones((5,5,5), dtype=bool, order="F")

    approximate_area = 5 * 5

    areafn = lambda n: xs3d.cross_sectional_area(labels, [2,2,2], n)

    assert areafn([1,0,0]) == approximate_area
    assert areafn([-1,0,0]) == approximate_area
    assert areafn([0,1,0]) == approximate_area
    assert areafn([0,-1,0]) == approximate_area
    assert areafn([0,0,1]) == approximate_area
    assert areafn([0,0,-1]) == approximate_area

    approximate_area = 5 * 5 * np.sqrt(2)

    assert np.isclose(areafn([1,1,0]), approximate_area)
    assert np.isclose(areafn([1,0,1]), approximate_area)
    assert np.isclose(areafn([0,1,1]), approximate_area)
    assert np.isclose(areafn([-1,-1,0]), approximate_area)
    assert np.isclose(areafn([-1,0,-1]), approximate_area)
    assert np.isclose(areafn([0,-1,-1]), approximate_area)
    
    assert np.isclose(areafn([-1,1,0]), approximate_area)
    assert np.isclose(areafn([1,-1,0]), approximate_area)
    assert np.isclose(areafn([0, 1,-1]), approximate_area)
    assert np.isclose(areafn([0,-1, 1]), approximate_area)

def test_empty():
    labels = np.zeros([0,0,0], dtype=bool)

    area = xs3d.cross_sectional_area(labels, [0,0,0], [1,1,1])
    assert area == 0


@pytest.mark.parametrize("off", [50, 25])
@pytest.mark.parametrize("normal", [[1,0,0], [0,1,0], [0,0,1], [1,1,1], [-1,-1,1], [.3,-.2,.7]])
def test_moving_window(off, normal):
    labels = np.zeros([100,100,100], dtype=bool, order="F")
    labels[:off, :off, :off] = True
    initial_area = xs3d.cross_sectional_area(labels, [off//2, off//2, off//2], normal)
    initial_area_slow = xs3d.cross_sectional_area(labels, [off//2, off//2, off//2], normal, slow_method=True)

    xs = xs3d.cross_section(labels, [off//2, off//2, off//2], normal, method=0)
    xs2 = xs3d.cross_section(labels, [off//2, off//2, off//2], normal, method=1)

    assert np.isclose(initial_area, initial_area_slow)
    assert np.isclose(xs.sum(), xs2.sum())

    for i in range(30):
        labels[:] = False
        labels[i:i+off, i:i+off, i:i+off] = True

        area = xs3d.cross_sectional_area(labels, [i+off//2, i+off//2, i+off//2], normal)
        assert np.isclose(area, initial_area)


@pytest.mark.parametrize("method", [0,1])
def test_cross_section(method):
    labels = np.ones((5,5,5), dtype=bool, order="F")
    pos = (2, 2, 2)

    def angle(theta):
        return [ 0, np.cos(theta), np.sin(theta) ]

    for theta in range(0,25):
        normal = angle(theta / 25 * 2 * np.pi)
        area = xs3d.cross_sectional_area(labels, pos, normal)
        image = xs3d.cross_section(labels, pos, normal, method=method)
        assert image.dtype == np.float32
        assert np.isclose(image.sum(), area)

def test_slice():
    labels = np.arange(9, dtype=np.uint8).reshape([3,3,1], order="F")
    slc = xs3d.slice(labels, [0,0,0], [0,0,1], standardize_basis=True)
    assert np.all(slc == labels[:,:,0])

    with pytest.raises(ValueError):
        area = xs3d.slice(labels, [0,0,0], [0,0,0])

    labels = np.ones([3,3,3,3], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.slice(labels, [0,0,0], [0,0,1])

def test_cross_sectional_area_inputs():
    labels = np.arange(9, dtype=np.uint8).reshape([3,3,1], order="F")

    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1])

    labels = np.ones([3,3,1], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,0])

    labels = np.ones([3,3,1], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1], [-1,1,1])
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1], [1,-1,1])
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1], [1,1,-1])
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1], [-1,-1,-1])

    labels = np.ones([3,3,3,3], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_sectional_area(labels, [0,0,0], [0,0,1], [1,1,1])

def test_cross_section_inputs():
    labels = np.arange(9, dtype=np.uint8).reshape([3,3,1], order="F")

    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1])

    labels = np.ones([3,3,1], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,0])

    labels = np.ones([3,3,1], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1], [-1,1,1])
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1], [1,-1,1])
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1], [1,1,-1])
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1], [-1,-1,-1])

    labels = np.ones([3,3,3,3], dtype=bool, order="F")
    with pytest.raises(ValueError):
        area = xs3d.cross_section(labels, [0,0,0], [0,0,1], [1,1,1])


def test_2d():
    labels = np.ones([3,3], dtype=bool)
    area = xs3d.cross_sectional_area(labels, [1,1], [0,1])
    assert area == 3

    area = xs3d.cross_sectional_area(labels, [1,1], [0,1], [3,3])
    assert area == 9

    area = xs3d.cross_sectional_area(labels, [1,1], [1,0], [1,1])
    assert area == 3

    area = xs3d.cross_sectional_area(labels, [1,1], [1,0], [5,5])
    assert area == 15

    area = xs3d.cross_sectional_area(labels, [0,0], [-1,1], [1,1])
    assert np.isclose(area, 3 * np.sqrt(2))

    area = xs3d.cross_sectional_area(labels, [-1,0], [-1,1], [1,1])
    assert area == 0

    area = xs3d.cross_sectional_area(labels, [1,-1], [-1,1], [1,1])
    assert area == 0

    labels[1,1] = 0
    area = xs3d.cross_sectional_area(labels, [1,1], [0,1], [1,1])
    assert area == 0


def test_off_axis_all_counted():
    arr = np.zeros([10,10,10], dtype=bool, order="F")
    arr[:5,:5,:5] = 1

    point = [2,2,2]
    normal = [1,1,1]

    fast_result = xs3d.cross_sectional_area(arr, point, normal)
    slow_result = xs3d.cross_sectional_area(arr, point, normal, slow_method=True)

    assert np.isclose(fast_result, slow_result)

def test_contact():
    labels = np.zeros([11,11,11], dtype=bool)
    labels[1:-1, 1:-1, 1:-1] = 1
    point = [5,5,5]
    normal = [0,0,1]

    result, contact = xs3d.cross_sectional_area(labels, point, normal, return_contact=True)
    assert contact == 0b00000000

    labels = np.ones([11,11,11], dtype=bool)
    result, contact = xs3d.cross_sectional_area(labels, point, normal, return_contact=True)
    assert contact == 0b00001111

    result, contact = xs3d.cross_sectional_area(labels, point, [0,1,0], return_contact=True)
    assert contact == 0b00110011

    result, contact = xs3d.cross_sectional_area(labels, point, [1,0,0], return_contact=True)
    assert contact == 0b00111100

    result, contact = xs3d.cross_sectional_area(labels, point, [1,1,1], return_contact=True)
    assert contact == 0b00111111

    result, contact = xs3d.cross_sectional_area(labels, [5,5,2], [1,1,1], return_contact=True)
    assert contact == 0b00111111

    result, contact = xs3d.cross_sectional_area(labels, [100,5,2], [1,1,1], return_contact=True)
    assert contact == 0

    labels = np.zeros([11,11,11], dtype=bool)
    labels[1:-1, :, :] = 1

    result, contact = xs3d.cross_sectional_area(labels, [5,5,5], [1,1,1], return_contact=True)
    assert contact == 0b00111100







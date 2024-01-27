#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include "xs3d.hpp"

namespace py = pybind11;

float xsa(
	const py::array_t<uint8_t> &binimg,
	const py::array_t<float> &point,
	const py::array_t<float> &normal,
	const py::array_t<float> &anisotropy
) {
	const uint64_t sx = binimg.shape()[0];
	const uint64_t sy = binimg.ndim() < 2
		? 1 
		: binimg.shape()[1];
	const uint64_t sz = binimg.ndim() < 3 
		? 1 
		: binimg.shape()[2];

	return xs3d::cross_sectional_area(
		binimg.data(),
		sx, sy, sz,
		point.at(0), point.at(1), point.at(2),
		normal.at(0), normal.at(1), normal.at(2),
		anisotropy.at(0), anisotropy.at(1), anisotropy.at(2)
	);
}

PYBIND11_MODULE(fastxs3d, m) {
	m.doc() = "Finding cross sectional area in 3D voxelized images."; 
	m.def("xsa", &xsa, "Find the cross sectional area for a given binary image, point, and normal vector.");
}

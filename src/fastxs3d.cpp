#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include "xs3d.hpp"

namespace py = pybind11;

auto section(
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

	const uint64_t voxels = sx * sy * sz;

	py::array_t arr = py::array_t<float, py::array::f_style>(voxels);
    float* data = static_cast<float*>(arr.request().ptr);
    std::fill(data, data + voxels, 0.0f);

	xs3d::cross_section(
		binimg.data(),
		sx, sy, sz,
		point.at(0), point.at(1), point.at(2),
		normal.at(0), normal.at(1), normal.at(2),
		anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
		data
	);

	return arr.reshape({ sx, sy, sz });
}

auto area(
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

	uint8_t contact = false;
	float area = xs3d::cross_sectional_area(
		binimg.data(),
		sx, sy, sz,
		point.at(0), point.at(1), point.at(2),
		normal.at(0), normal.at(1), normal.at(2),
		anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
		contact
	);
	return std::tuple(area, contact);
}

auto projection(	
	const py::array &labels,
	const py::array_t<float> &point,
	const py::array_t<float> &normal
) {
	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.ndim() < 2
		? 1 
		: labels.shape()[1];
	const uint64_t sz = labels.ndim() < 3 
		? 1 
		: labels.shape()[2];

	uint64_t psx = 2 * sqrt(3) * std::max(std::max(sx,sy), sz) + 1;
	uint64_t pvoxels = psx * psx;

	py::array arr; 

	auto projectionfn = [&](auto dtype) {
		arr = py::array_t<decltype(dtype), py::array::f_style>(pvoxels);
		auto out = reinterpret_cast<decltype(dtype)*>(arr.request().ptr);
		auto data = reinterpret_cast<decltype(dtype)*>(labels.request().ptr);
		std::fill(out, out + pvoxels, 0);

		xs3d::cross_section_projection<decltype(dtype)>(
			data,
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			out
		);
	};

	int data_width = labels.dtype().itemsize();

    if (data_width == 1) {
    	projectionfn(uint8_t{});
    }
    else if (data_width == 2) {
    	projectionfn(uint16_t{});
    }
    else if (data_width == 4) {
    	projectionfn(uint32_t{});
    }
    else if (data_width == 8) {
    	projectionfn(uint64_t{});
    }
    else {
    	throw new std::runtime_error("should never happen");
    }

	return arr.reshape({ psx, psx }).view(py::str(labels.dtype()));
}

PYBIND11_MODULE(fastxs3d, m) {
	m.doc() = "Finding cross sectional area in 3D voxelized images."; 
	m.def("projection", &projection, "Project a cross section of a 3D image onto a 2D plane");
	m.def("section", &section, "Return a binary image that highlights the voxels contributing area to a cross section.");
	m.def("area", &area, "Find the cross sectional area for a given binary image, point, and normal vector.");
}

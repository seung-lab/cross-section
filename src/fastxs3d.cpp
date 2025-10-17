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
	const py::array_t<float> &anisotropy,
	const int method = 0
) {
	const uint64_t sx = binimg.shape()[0];
	const uint64_t sy = binimg.ndim() < 2
		? 1 
		: binimg.shape()[1];
	const uint64_t sz = binimg.ndim() < 3 
		? 1 
		: binimg.shape()[2];

	const uint64_t voxels = sx * sy * sz;

	uint64_t vsx = sx;
	uint64_t vsy = sy;
	uint64_t vsz = sz;
	uint64_t vvoxels = voxels;

	if (method >= 2) {
		vsx = (sx+1) >> 1;
		vsy = (sy+1) >> 1;
		vsz = (sz+1) >> 1;
		vvoxels = vsx * vsy * vsz;
	}

	py::array_t arr = py::array_t<float, py::array::f_style>({ vsx, vsy, vsz });
    float* data = static_cast<float*>(arr.request().ptr);
    std::fill(data, data + vvoxels, 0.0f);

	std::tuple<float*, uint8_t> tup;

	if (method == 0) {
		 tup = xs3d::cross_section(
			binimg.data(),
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
			data
		);
	}
	else if (method == 1) {
		tup = xs3d::cross_section_slow(
			binimg.data(),
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
			data
		);
	}
	else {
		 tup = xs3d::cross_section_slow_2x2x2(
			binimg.data(),
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
			data
		);
	}

	return std::make_tuple(arr, std::get<1>(tup));
}

template <typename LABEL>
auto calculate_area(
	const py::array_t<LABEL> &image,
	const LABEL segid,
	const py::array_t<float> &point,
	const py::array_t<float> &normal,
	const py::array_t<float> &anisotropy,
	const bool slow_method,
	const bool use_persistent_data
) {
	const uint64_t sx = image.shape()[0];
	const uint64_t sy = image.ndim() < 2
		? 1 
		: image.shape()[1];
	const uint64_t sz = image.ndim() < 3 
		? 1 
		: image.shape()[2];

	if (point.size() < 3) {
	    throw py::value_error("point array must have at least 3 elements");
	}

	if (normal.size() < 3) {
	    throw py::value_error("normal array must have at least 3 elements");
	}

	if (anisotropy.size() < 3) {
	    throw py::value_error("anisotropy array must have at least 3 elements");
	}

	if (sx == 0 || sy == 0 || sz == 0) {
	    return std::make_tuple(static_cast<float>(0.0), static_cast<uint8_t>(0));
	}

	if (slow_method) {
		return xs3d::cross_sectional_area_slow<LABEL>(
			image.data(), segid,
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2)
		);
	}
	else {
		return xs3d::cross_sectional_area<LABEL>(
			image.data(), segid,
			sx, sy, sz,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
			use_persistent_data
		);
	}
}

auto projection(	
	const py::array &labels,
	const py::array_t<float> &point,
	const py::array_t<float> &normal,
	const py::array_t<float> &anisotropy,
	const bool standardize_basis,
	const float crop_distance
) {
	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.ndim() < 2
		? 1 
		: labels.shape()[1];
	const uint64_t sz = labels.ndim() < 3 
		? 1 
		: labels.shape()[2];

	const bool c_order = py::array::c_style == (labels.flags() & py::array::c_style);

	float wx = anisotropy.at(0);
	float wy = anisotropy.at(1);
	float wz = anisotropy.at(2);

	float minval = std::min(std::min(wx,wy), wz);
	wx /= minval; wy /= minval; wz /= minval;
	float maxval = std::max(std::max(std::abs(wx), std::abs(wy)), std::abs(wz));

	const uint64_t distortion = static_cast<uint64_t>(ceil(maxval));

	// rational approximation of sqrt(3) is 97/56
	// result is more likely to be same across compilers
	uint64_t largest_dimension = std::max(std::max(sx,sy), sz);
	if (static_cast<float>(largest_dimension) > crop_distance && crop_distance >= 0) {
		largest_dimension = static_cast<uint64_t>(std::ceil(crop_distance));
	}

	const uint64_t psx = (distortion * 2 * 97 * largest_dimension / 56) + 1;
	const uint64_t pvoxels = psx * psx;

	auto projectionfn = [&](auto dtype) {
		auto cutout = py::array_t<decltype(dtype), py::array::f_style>(std::vector<int64_t>{ 0, 0 });

		if (crop_distance == 0) {
			return cutout.view(py::str(labels.dtype()));
		}

		py::array arr = py::array_t<decltype(dtype), py::array::f_style>({ psx, psx });
		auto out = reinterpret_cast<decltype(dtype)*>(arr.request().ptr);
		auto data = reinterpret_cast<decltype(dtype)*>(labels.request().ptr);
		std::fill(out, out + pvoxels, 0);

		std::tuple<decltype(dtype)*, xs3d::Bbox2d> tup = xs3d::cross_section_projection<decltype(dtype)>(
			data,
			sx, sy, sz, c_order,
			point.at(0), point.at(1), point.at(2),
			normal.at(0), normal.at(1), normal.at(2),
			anisotropy.at(0), anisotropy.at(1), anisotropy.at(2),
			standardize_basis, crop_distance,
			out
		);

		xs3d::Bbox2d bbox = std::get<1>(tup);
		bbox.x_max++;
		bbox.y_max++;

		cutout = py::array_t<decltype(dtype), py::array::f_style>(std::vector<int64_t>{ bbox.sx(), bbox.sy() });
	    auto cutout_ptr = reinterpret_cast<decltype(dtype)*>(cutout.request().ptr);

	    int64_t csx = bbox.sx();

	    for (int64_t y = bbox.y_min; y < bbox.y_max; y++) {
	        for (int64_t x = bbox.x_min; x < bbox.x_max; x++) {
	            cutout_ptr[
	            	(x - bbox.x_min) + csx * (y - bbox.y_min)
	            ] = out[x + psx * y];
	        }
	    }
	    
		return cutout.view(py::str(labels.dtype()));
	};

	int data_width = labels.dtype().itemsize();

    if (data_width == 1) {
    	return projectionfn(uint8_t{});
    }
    else if (data_width == 2) {
    	return projectionfn(uint16_t{});
    }
    else if (data_width == 4) {
    	return projectionfn(uint32_t{});
    }
    else if (data_width == 8) {
    	return projectionfn(uint64_t{});
    }
    else {
    	throw new std::runtime_error("should never happen");
    }
}


#define REGISTER_CALCULATE_AREA_FOR_TYPE(T) \
    m.def("area", &calculate_area<T>, "Find the cross sectional area for a given image, point, and normal vector.");

PYBIND11_MODULE(fastxs3d, m) {
	m.doc() = "Finding cross sectional area in 3D voxelized images."; 
	m.def("projection", &projection, "Project a cross section of a 3D image onto a 2D plane");
	m.def("section", &section, "Return a floating point image that shows the voxels contributing area to a cross section.");

	REGISTER_CALCULATE_AREA_FOR_TYPE(bool)
	REGISTER_CALCULATE_AREA_FOR_TYPE(uint8_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(uint16_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(uint32_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(uint64_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(int8_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(int16_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(int32_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(int64_t)
	REGISTER_CALCULATE_AREA_FOR_TYPE(float)
	REGISTER_CALCULATE_AREA_FOR_TYPE(double)
	
	m.def("set_shape", &xs3d::set_shape, "Accelerate the area function across many evaluation points by saving some attributes of the input shape upfront. Call clear_shape when you are done.");
	m.def("clear_shape", &xs3d::clear_shape, "Delete the data that was persisted by set_shape.");
}

#undef REGISTER_CALCULATE_AREA_FOR_TYPE
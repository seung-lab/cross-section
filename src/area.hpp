#ifndef __XS3D_AREA_HPP__
#define __XS3D_AREA_HPP__

#include <array>
#include <vector>
#include <utility>

#include "vec.hpp"

using namespace xs3d;

namespace {

float area_of_triangle(
	const std::vector<Vec3>& pts, 
	const Vec3& anisotropy
) {
	Vec3 v1 = pts[1] - pts[0];
	v1 *= anisotropy;
	Vec3 v2 = pts[2] - pts[0];
	v2 *= anisotropy;
	Vec3 v3 = v1.cross(v2);
	return v3.norm() * 0.5;
}

#define CMP_SWAP(x,y) \
	if (values[x] > values[y]) {\
		std::swap(values[x], values[y]);\
		std::swap(vecs[x], vecs[y]);\
	}

#define CMP_SWAP_FAST(x,y) \
	if (values[x] > values[y]) {\
		std::swap(vecs[x], vecs[y]);\
	}

template <size_t N>
inline void calculate_sort_projections(
	std::array<float, N>& values,
	std::vector<Vec3>& vecs,
	const Vec3& prime_spoke,
	const Vec3& basis
) {
	#pragma unroll
	for (uint64_t i = 0; i < N; i++) {
		Vec3& vec = vecs[i];
		float projection = vec.dot(prime_spoke) / vec.norm();
		values[i] = (vec.dot(basis) < 0) 
			? (projection - 1) 
			: (1 - projection);
	}
}

template <size_t N>
void sorting_network(
	std::vector<Vec3>& vecs,
	const Vec3& prime_spoke,
	const Vec3& basis
);

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,2),(1,3)]
[(0,1),(2,3)]
[(1,2)]
*/
template <>
void sorting_network<4>(
	std::vector<Vec3>& vecs,
	const Vec3& prime_spoke,
	const Vec3& basis
) {
	static thread_local std::array<float, 4> values = {};
	calculate_sort_projections<4>(values, vecs, prime_spoke, basis);

	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP_FAST(1,2)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,3),(1,4)]
[(0,2),(1,3)]
[(0,1),(2,4)]
[(1,2),(3,4)]
[(2,3)] 
*/
template <>
void sorting_network<5>(
	std::vector<Vec3>& vecs,
	const Vec3& prime_spoke,
	const Vec3& basis
) {
	static thread_local std::array<float, 5> values = {};
	calculate_sort_projections<5>(values, vecs, prime_spoke, basis);

	CMP_SWAP(0,3)
	CMP_SWAP(1,4)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(0,1)
	CMP_SWAP(2,4)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP_FAST(2,3)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,5),(1,3),(2,4)]
[(1,2),(3,4)]
[(0,3),(2,5)]
[(0,1),(2,3),(4,5)]
[(1,2),(3,4)]
*/
template <>
void sorting_network<6>(
	std::vector<Vec3>& vecs,
	const Vec3& prime_spoke,
	const Vec3& basis
) {
	static thread_local std::array<float, 6> values = {};
	calculate_sort_projections<6>(values, vecs, prime_spoke, basis);

	CMP_SWAP(0,5)
	CMP_SWAP(1,3)
	CMP_SWAP(2,4)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(0,3)
	CMP_SWAP(2,5)
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP_FAST(1,2)
	CMP_SWAP_FAST(3,4)
}

#undef CMP_SWAP

template <size_t N>
float spokes_to_area(
	std::vector<Vec3>& spokes,
	const Vec3& anisotropy
) {
	#pragma unroll
	for (uint64_t i = 0; i < N; i++) {
		spokes[i] *= anisotropy;
	}

	float area = 0.0;
	#pragma unroll
	for (uint64_t i = 0; i < N - 1; i++) {
		area += spokes[i].cross(spokes[i+1]).norm();
	}
	area += spokes[0].cross(spokes[N - 1]).norm();

	return area * 0.5;
}

float area_of_poly(
	const std::vector<Vec3>& pts, 
	const Vec3& normal,
	const Vec3& anisotropy
) {
	
	Vec3 centroid(0,0,0);
	const size_t N_pts = pts.size();

	for (Vec3 pt : pts) {
		centroid += pt;
	}
	centroid /= static_cast<float>(N_pts);

	static thread_local std::vector<Vec3> spokes(6);
	
	for (size_t i = 0; i < N_pts; i++) {
		spokes[i] = (pts[i] - centroid);
	}

	Vec3 prime_spoke = spokes[0];
	prime_spoke /= prime_spoke.norm();

	Vec3 basis = prime_spoke.cross(normal);
	basis /= basis.norm();

	if (N_pts == 4) {
		sorting_network<4>(spokes, prime_spoke, basis);
		return spokes_to_area<4>(spokes, anisotropy);
	}
	else if (N_pts == 5) {
		sorting_network<5>(spokes, prime_spoke, basis);
		return spokes_to_area<5>(spokes, anisotropy);
	}
	else {
		sorting_network<6>(spokes, prime_spoke, basis);
		return spokes_to_area<6>(spokes, anisotropy);
	}
}

};

namespace xs3d::area {

float points_to_area(
	const std::vector<Vec3>& pts,
	const Vec3& anisotropy,
	const Vec3& normal
) {
	const size_t size = pts.size();

	if (size < 3) {
		// no contact, point, or line which have zero area
		return 0;
	}
	else if (size == 3) {
		return area_of_triangle(pts, anisotropy);
	}
	else { // 4, 5, 6
		return area_of_poly(pts, normal, anisotropy);
	}
}

};

#endif 

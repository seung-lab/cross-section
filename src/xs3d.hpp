#ifndef __XS3D_HPP__
#define __XS3D_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stack>
#include <utility>
#include <vector>
#include <stdexcept>

#include "vec.hpp"
#include "area.hpp"
#include "builtins.hpp"

using namespace xs3d;

namespace {

// half rounded up
uint64_t _h(uint64_t a) { 
	return ((a+1) >> 1); 
};

template <typename LABEL>
uint8_t compute_cube(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t x, const uint64_t y, const uint64_t z
) {
	const uint64_t sxy = sx * sy;
	const uint64_t loc = x + sx * (y + sy * z);

    const uint64_t x_valid = (x < sx - 1);
    const uint64_t y_valid = (y < sy - 1);
    const uint64_t z_valid = (z < sz - 1);

	return static_cast<uint8_t>(
		(labels[loc] == segid)
		| ((x_valid && (labels[loc+1] == segid)) << 1)
		| ((y_valid && (labels[loc+sx] == segid)) << 2)
		| (((x_valid && y_valid) && (labels[loc+sx+1] == segid)) << 3)
		| ((z_valid && (labels[loc+sxy] == segid)) << 4)
		| (((x_valid && z_valid) && (labels[loc+sxy+1] == segid)) << 5)
		| (((y_valid && z_valid) && (labels[loc+sxy+sx] == segid)) << 6)
		| (((x_valid && y_valid && z_valid) && (labels[loc+sxy+sx+1] == segid)) << 7)
	);
}

struct PersistentShapeManager {
	std::vector<uint8_t> visited;
	uint64_t sx, sy, sz;
	uint8_t color;

	PersistentShapeManager() {
		sx = 0; sy = 0; sz = 0;
		color = 0;
	}

	void init(const uint64_t _sx, const uint64_t _sy, const uint64_t _sz) {
		sx = _sx;
		sy = _sy;
		sz = _sz;
		visited.resize(this->eighth_voxels());
	}

	void maybe_grow(const uint64_t _sx, const uint64_t _sy, const uint64_t _sz) {
		if (sx * sy * sz < _sx * _sy * _sz) {
			init(_sx, _sy, _sz);
		}
	}

	void clear() {
		sx = 0;
		sy = 0;
		sz = 0;
		color = 0;
		visited = std::vector<uint8_t>();
	}

	uint64_t eighth_voxels() {
		return _h(sx) * _h(sy) * _h(sz);
	}

	uint8_t next_color() {
		color++;
		if (color == 0) {
			std::fill(visited.begin(), visited.end(), 0);
			color = 1;
		}

		return color;
	}
};

PersistentShapeManager persistent_data;

const Vec3 ihat = Vec3(1,0,0);
const Vec3 jhat = Vec3(0,1,0);
const Vec3 khat = Vec3(0,0,1);

uint16_t check_intersections_2x2x2(
	std::vector<Vec3>& pts,
	const uint64_t x, const uint64_t y, const uint64_t z,
	const Vec3& pos, const Vec3& normal, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {
	static const Vec3 c[8] = {
		Vec3(0, 0, 0), // 0
		Vec3(0, 0, 2), // 1 
		Vec3(0, 2, 0), // 2 
		Vec3(0, 2, 2), // 3
		Vec3(2, 0, 0), // 4
		Vec3(2, 0, 2), // 5
		Vec3(2, 2, 0), // 6
		Vec3(2, 2, 2) // 7
	};

	static const Vec3 pipes[3] = {
		ihat, jhat, khat
	};

	static const Vec3 pipe_points[4] = {
		c[0], c[3], c[5], c[6]
	};

	pts.clear();

	Vec3 centerpt(x,y,z);
	centerpt += 0.5;

	// for testing the 2x2x2 field, we need to move the point
	// to the center of the grid. then if the distance to the plane is
	// > sqrt(3), it's not intersecting.

	constexpr float epsilon = 2e-5;
	constexpr float max_dist_to_plane = 1.7320508076 + epsilon;

	float dist_to_plane = std::abs((centerpt-pos).dot(normal));
	// if the distance to the plane is greater than sqrt(3)/2
	// then the plane is not intersecting at all.
	if (dist_to_plane > max_dist_to_plane) { 
		return 0;
	}

	Vec3 minpt(x,y,z);
	minpt -= 0.5;

	Vec3 pos2 = pos - minpt;

	float corner_projections[4] = {
		(pos2 - c[0]).dot(normal),
		(pos2 - c[3]).dot(normal),
		(pos2 - c[5]).dot(normal),
		(pos2 - c[6]).dot(normal),
	};

	auto inlist = [&](const Vec3& pt){
		for (const Vec3& p : pts) {
			if (p.close(pt)) {
				return true;
			}
		}
		return false;
	};

	const uint64_t max_pts = (normal.num_zero_dims() >= 1)
		? 4
		: 6;

	constexpr float bound = 1 + epsilon;

	Vec3 corner;
	uint16_t edges = 0;

	// edges is marked with
	// iteration order (i) as:
	// 000x, 022x, 202x, 220x
	// 000y, 022y, 202y, 220y
	// 000, 022z, 202z, 220z

	for (int i = 0; i < 12; i++) {		
		float proj = corner_projections[i & 0b11];

		if (proj == 0) {
			corner = pipe_points[i & 0b11];
			corner += minpt;
			if (i < 4 && !inlist(corner)) {
				pts.push_back(corner);
			}
			continue;
		}

		// if traveling parallel to plane but
		// not on the plane
		if (projections[i >> 2] == 0) {
			continue;
		}

		float t = proj * inv_projections[i >> 2];
		if (std::abs(t) > 2 + epsilon) {
			continue;
		}

		Vec3 pipe = pipes[i >> 2];
		corner = pipe_points[i & 0b11];
		corner += minpt;
		Vec3 nearest_pt = corner + pipe * t;

		if (std::abs(nearest_pt.x - centerpt.x) >= bound) {
			continue;
		}
		else if (std::abs(nearest_pt.y - centerpt.y) >= bound) {
			continue;
		}
		else if (std::abs(nearest_pt.z - centerpt.z) >= bound) {
			continue;
		}

		// if t = -2, 0, 2 we're on a corner, which are the only areas where a
		// duplicate vertex is possible.
		const bool iscorner = (std::abs(t) < epsilon || std::abs(std::abs(t)-2) < epsilon);

		if (!iscorner || !inlist(nearest_pt)) {
			pts.push_back(nearest_pt);
			edges |= (1 << i);

			if (pts.size() >= max_pts) {
				break;
			}
		}
	}

	return edges;
}

void check_intersections_1x1x1(
	std::vector<Vec3>& pts,
	const uint64_t x, const uint64_t y, const uint64_t z,
	const Vec3& pos, const Vec3& normal, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {

	static const Vec3 c[8] = {
		Vec3(0, 0, 0), // 0
		Vec3(0, 0, 1), // 1 
		Vec3(0, 1, 0), // 2 
		Vec3(0, 1, 1), // 3
		Vec3(1, 0, 0), // 4
		Vec3(1, 0, 1), // 5
		Vec3(1, 1, 0), // 6
		Vec3(1, 1, 1) // 7
	};

	static const Vec3 pipes[3] = {
		ihat, jhat, khat
	};

	static const Vec3 pipe_points[4] = {
		c[0], c[3], c[5], c[6]
	};

	pts.clear();

	Vec3 minpt(x,y,z);

	constexpr float epsilon = 2e-5;
	constexpr float max_dist_to_plane = 1.7320508076 / 2 + epsilon;

	float dist_to_plane = std::abs((minpt-pos).dot(normal));
	// if the distance to the plane is greater than sqrt(3)/2
	// then the plane is not intersecting at all.
	if (dist_to_plane > max_dist_to_plane) { 
		return;
	}

	minpt += -0.5;

	Vec3 pos2 = pos - minpt;

	float corner_projections[4] = {
		(pos2 - c[0]).dot(normal),
		(pos2 - c[3]).dot(normal),
		(pos2 - c[5]).dot(normal),
		(pos2 - c[6]).dot(normal),
	};

	auto inlist = [&](const Vec3& pt){
		for (const Vec3& p : pts) {
			if (p.close(pt)) {
				return true;
			}
		}
		return false;
	};

	const uint64_t max_pts = (normal.num_zero_dims() >= 1)
		? 4
		: 6;

	constexpr float bound = 0.5 + epsilon;

	Vec3 corner;

	for (int i = 0; i < 12; i++) {		
		float proj = corner_projections[i & 0b11];

		if (proj == 0) {
			corner = pipe_points[i & 0b11];
			corner += minpt;
			if (i < 4 && !inlist(corner)) {
				pts.push_back(corner);
			}
			continue;
		}

		float proj2 = projections[i >> 2];

		// if traveling parallel to plane but
		// not on the plane
		if (proj2 == 0) {
			continue;
		}

		float t = proj * inv_projections[i >> 2];
		if (std::abs(t) > 1 + epsilon) {
			continue;
		}

		Vec3 pipe = pipes[i >> 2];
		corner = pipe_points[i & 0b11];
		corner += minpt;
		Vec3 nearest_pt = corner + pipe * t;

		if (std::abs(nearest_pt.x - x) >= bound) {
			continue;
		}
		else if (std::abs(nearest_pt.y - y) >= bound) {
			continue;
		}
		else if (std::abs(nearest_pt.z - z) >= bound) {
			continue;
		}

		// if t = -1, 0, 1 we're on a corner, which are the only areas where a
		// duplicate vertex is possible.
		const bool iscorner = (std::abs(t) < epsilon || std::abs(std::abs(t)-1) < epsilon);

		if (!iscorner || !inlist(nearest_pt)) {
			pts.push_back(nearest_pt);

			if (pts.size() >= max_pts) {
				break;
			}
		}
	}
}

float calc_area_at_point_2x2x2(
	uint8_t cube,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& cur, const Vec3& pos, 
	const Vec3& normal, const Vec3& anisotropy,
	std::vector<Vec3>& pts, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {
	const uint64_t x = static_cast<uint64_t>(cur.x) & ~1;
	const uint64_t y = static_cast<uint64_t>(cur.y) & ~1;
	const uint64_t z = static_cast<uint64_t>(cur.z) & ~1;

	Vec3 centerpt(x,y,z);
	centerpt += 0.5;

	// for testing the 2x2x2 field, we need to move the point
	// to the center of the grid. then if the distance to the plane is
	// > sqrt(3), it's not intersecting.

	constexpr float epsilon = 2e-5;
	constexpr float max_dist_to_plane = 1.7320508076 + epsilon;

	float dist_to_plane = std::abs((centerpt-pos).dot(normal));
	// if the distance to the plane is greater than sqrt(3)/2
	// then the plane is not intersecting at all.
	if (dist_to_plane > max_dist_to_plane) { 
		return 0.0;
	}

	auto areafn2 = [&]() {
		check_intersections_2x2x2(
			pts, 
			x, y, z,
			pos, normal, 
			projections, inv_projections
		);

		return xs3d::area::points_to_area(pts, anisotropy, normal);
	};

	auto areafn1 = [&](uint8_t idx) {
		const uint64_t oz = idx >> 2;
		const uint64_t oy = (idx - (oz << 2)) >> 1;
		const uint64_t ox = (idx - (oz << 2) - (oy << 1));

		check_intersections_1x1x1(
			pts, 
			x+ox, y+oy, z+oz,
			pos, normal, 
			projections, inv_projections
		);

		return xs3d::area::points_to_area(pts, anisotropy, normal);
	};

	float area = 0;
	uint8_t idx = 0;

	if (popcount(cube) >= 5) {
		area = areafn2();
		if (area == 0) {
			return area;
		}
		while (static_cast<uint8_t>(~cube)) {
			idx = ffs(~cube) - 1;
			area -= areafn1(idx);
			cube |= (1 << idx);
		}
	}
	else {
		while (static_cast<uint8_t>(cube)) {
			idx = ffs(cube) - 1;
			area += areafn1(idx);
			cube = cube & ~(1 << idx);
		}
	}

	return area;
}

template <typename LABEL>
float calc_area_at_point(
	const LABEL* labels, const LABEL segid,
	std::vector<bool>& ccl,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& cur, const Vec3& pos, 
	const Vec3& normal, const Vec3& anisotropy,
	std::vector<Vec3>& pts, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections,
	float* plane_visualization
) {

	const uint64_t voxels = sx * sy * sz;

	float subtotal = 0.0;

	float xs = (cur.x - 1) >= 0 ? -1 : 0;
	float ys = (cur.y - 1) >= 0 ? -1 : 0;
	float zs = (cur.z - 1) >= 0 ? -1 : 0;

	float xe = (cur.x + 1) < sx ? 1 : 0;
	float ye = (cur.y + 1) < sy ? 1 : 0;
	float ze = (cur.z + 1) < sz ? 1 : 0;
	
	// only need to check around the current voxel if
	// there's a possibility that there is a gap due
	// to basis vector motion. If the normal is axis
	// aligned to x, y, or z, there will be no gap.
	if (normal.is_axis_aligned()) {
		xs = 0;
		ys = 0;
		zs = 0;

		xe = 0;
		ye = 0;
		ze = 0;		
	}

	for (float z = zs; z <= ze; z++) {
		for (float y = ys; y <= ye; y++) {
			for (float x = xs; x <= xe; x++) {
				
				Vec3 delta(x,y,z);
				delta += cur;

				// boundaries between voxels are located at 0.5
				delta.x = std::round(delta.x);
				delta.y = std::round(delta.y);
				delta.z = std::round(delta.z);

				if (
					   delta.x < 0 || delta.x >= sx 
					|| delta.y < 0 || delta.y >= sy 
					|| delta.z < 0 || delta.z >= sz
				) {
					continue;
				}

				uint64_t loc = static_cast<uint64_t>(delta.x) + sx * (
					static_cast<uint64_t>(delta.y) + sy * static_cast<uint64_t>(delta.z)
				);

				if (loc < 0 || loc >= voxels) {
					continue;
				}
				else if (labels[loc] != segid) {
					continue;
				}

				if (ccl[loc] == 0) {
					ccl[loc] = 1;
					
					check_intersections_1x1x1(
						pts, 
						static_cast<uint64_t>(delta.x), 
						static_cast<uint64_t>(delta.y), 
						static_cast<uint64_t>(delta.z),
						pos, normal, 
						projections, inv_projections
					);

					const float area = xs3d::area::points_to_area(pts, anisotropy, normal);
					subtotal += area;

					if (plane_visualization != NULL && area > 0.0) {
						plane_visualization[loc] = area;
					}
				}
			}
		}
	}

	return subtotal;
}

bool is_26_connected(
	const uint8_t center, const uint8_t candidate, 
	const int x, const int y, const int z
) {
	if (x < 0) {
		if (y < 0) {
			if (z < 0) {
				return ((candidate & 0b10000000) > 0) & ((center & 0b00000001) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b10001000) > 0) & ((center & 0b00010001) > 0);
			}
			else {
				return ((candidate & 0b00001000) > 0) & ((center & 0b00010000) > 0);
			}
		}
		else if (y == 0) {
			if (z < 0) {
				return ((candidate & 0b10100000) > 0) & ((center & 0b00000101) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b10101010) > 0) & ((center & 0b01010101) > 0);
			}
			else {
				return ((candidate & 0b00001010) > 0) & ((center & 0b01010000) > 0);
			}
		}
		else {
			if (z < 0) {
				return ((candidate & 0b00100000) > 0) & ((center & 0b00000100) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b00100010) > 0) & ((center & 0b01000010) > 0);
			}
			else {
				return ((candidate & 0b00000010) > 0) & ((center & 0b01000000) > 0);
			}
		}
	}
	else if (x == 0) {
		if (y < 0) {
			if (z < 0) {
				return ((candidate & 0b11000000) > 0) & ((center & 0b00000011) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b11001100) > 0) & ((center & 0b00110011) > 0);
			}
			else {
				return ((candidate & 0b00001100) > 0) & ((center & 0b00110000) > 0);
			}
		}
		else if (y == 0) {
			if (z < 0) {
				return ((candidate & 0b11110000) > 0) & ((center & 0b00001111) > 0);
			}
			else if (z == 0) {
				return true;
			}
 			else {
				return ((candidate & 0b00001111) > 0) & ((center & 0b11110000) > 0);
			}
		}
		else {
			if (z < 0) {
				return ((candidate & 0b00110000) > 0) & ((center & 0b00001100) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b00110011) > 0) & ((center & 0b11001100) > 0);
			}
			else {
				return ((candidate & 0b00000011) > 0) & ((center & 0b11000000) > 0);
			}
		}
	}
	else {
		if (y < 0) {
			if (z < 0) {
				return ((candidate & 0b01000000) > 0) & ((center & 0b00000010) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b01000100) > 0) & ((center & 0b00100010) > 0);
			}
			else {
				return ((candidate & 0b00000100) > 0) & ((center & 0b00100000) > 0);
			}
		}
		else if (y == 0) {
			if (z < 0) {
				return ((candidate & 0b01010000) > 0) & ((center & 0b00001010) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b01010101) > 0) & ((center & 0b10101010) > 0);
			}
			else {
				return ((candidate & 0b00000101) > 0) & ((center & 0b10100000) > 0);
			}
		}
		else {
			if (z < 0) {
				return ((candidate & 0b00010000) > 0) & ((center & 0b00001000) > 0);
			}
			else if (z == 0) {
				return ((candidate & 0b00010001) > 0) & ((center & 0b10001000) > 0);
			}
			else {
				return ((candidate & 0b00000001) > 0) & ((center & 0b10000000) > 0);
			}
		}
	}
}

template <typename LABEL>
float robust_calc_area_at_point_2x2x2(
	const LABEL* labels, const LABEL segid,
	std::vector<bool>& ccl,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& cur, const Vec3& pos, 
	const Vec3& normal, const Vec3& anisotropy,
	std::vector<Vec3>& pts, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {

	uint64_t x = static_cast<uint64_t>(cur.x) & ~static_cast<uint64_t>(1);
	uint64_t y = static_cast<uint64_t>(cur.y) & ~static_cast<uint64_t>(1);
	uint64_t z = static_cast<uint64_t>(cur.z) & ~static_cast<uint64_t>(1);

	float subtotal = 0.0;

	float xs = (cur.x - 2) >= 0 ? -2 : 0;
	float ys = (cur.y - 2) >= 0 ? -2 : 0;
	float zs = (cur.z - 2) >= 0 ? -2 : 0;

	float xe = (cur.x + 2) < sx ? 2 : 0;
	float ye = (cur.y + 2) < sy ? 2 : 0;
	float ze = (cur.z + 2) < sz ? 2 : 0;
	
	// only need to check around the current voxel if
	// there's a possibility that there is a gap due
	// to basis vector motion. If the normal is axis
	// aligned to x, y, or z, there will be no gap.
	if (normal.is_axis_aligned()) {
		xs = 0;
		ys = 0;
		zs = 0;

		xe = 0;
		ye = 0;
		ze = 0;		
	}

	const uint8_t center = compute_cube(labels, segid, sx, sy, sz, x, y, z);

	for (int64_t zi = zs; zi <= ze; zi += 2) {
		for (int64_t yi = ys; yi <= ye; yi += 2) {
			for (int64_t xi = xs; xi <= xe; xi += 2) {
				
				Vec3 delta(xi,yi,zi);
				delta += cur;

				const uint64_t loc = static_cast<uint64_t>(delta.x) + sx * (
					static_cast<uint64_t>(delta.y) + sy * static_cast<uint64_t>(delta.z)
				);

				const uint64_t ccl_loc =  (static_cast<uint64_t>(delta.x) >> 1) + _h(sx) * (
					(static_cast<uint64_t>(delta.y) >> 1) + _h(sy) * (static_cast<uint64_t>(delta.z) >> 1)
				);

				if ((labels[loc] != segid) || ccl[ccl_loc]) {
					continue;
				}
				
				ccl[ccl_loc] = true;
					
				uint8_t candidate = compute_cube(labels, segid, sx, sy, sz, x + xi, y + yi, z + zi);

				if (is_26_connected(center, candidate, xi, yi, zi)) {
					subtotal += calc_area_at_point_2x2x2(
						candidate,
						sx, sy, sz,
						delta, pos, normal, anisotropy,
						pts, 
						projections, inv_projections
					);
				}
			}
		}
	}

	return subtotal;
}

template <typename LABEL>
float robust_calc_area_at_point_2x2x2_persistent_data(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& cur, const Vec3& pos, 
	const Vec3& normal, const Vec3& anisotropy,
	std::vector<Vec3>& pts, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {

	uint64_t x = static_cast<uint64_t>(cur.x) & ~static_cast<uint64_t>(1);
	uint64_t y = static_cast<uint64_t>(cur.y) & ~static_cast<uint64_t>(1);
	uint64_t z = static_cast<uint64_t>(cur.z) & ~static_cast<uint64_t>(1);

	float subtotal = 0.0;

	float xs = (cur.x - 2) >= 0 ? -2 : 0;
	float ys = (cur.y - 2) >= 0 ? -2 : 0;
	float zs = (cur.z - 2) >= 0 ? -2 : 0;

	float xe = (cur.x + 2) < sx ? 2 : 0;
	float ye = (cur.y + 2) < sy ? 2 : 0;
	float ze = (cur.z + 2) < sz ? 2 : 0;
	
	// only need to check around the current voxel if
	// there's a possibility that there is a gap due
	// to basis vector motion. If the normal is axis
	// aligned to x, y, or z, there will be no gap.
	if (normal.is_axis_aligned()) {
		xs = 0;
		ys = 0;
		zs = 0;

		xe = 0;
		ye = 0;
		ze = 0;		
	}

	const uint8_t center = compute_cube(labels, segid, sx, sy, sz, x, y, z);
	std::vector<uint8_t>& visited = persistent_data.visited;
	const uint8_t color = persistent_data.color;

	for (int64_t zi = zs; zi <= ze; zi += 2) {
		for (int64_t yi = ys; yi <= ye; yi += 2) {
			for (int64_t xi = xs; xi <= xe; xi += 2) {
				
				Vec3 delta(xi,yi,zi);
				delta += cur;

				const uint64_t loc = static_cast<uint64_t>(delta.x) + sx * (
					static_cast<uint64_t>(delta.y) + sy * static_cast<uint64_t>(delta.z)
				);

				const uint64_t visited_loc =  (static_cast<uint64_t>(delta.x) >> 1) + _h(sx) * (
					(static_cast<uint64_t>(delta.y) >> 1) + _h(sy) * (static_cast<uint64_t>(delta.z) >> 1)
				);

				if ((labels[loc] != segid) || visited[visited_loc] == color) {
					continue;
				}
				
				visited[visited_loc] = color;
				
				const uint8_t candidate = compute_cube(labels, segid, sx, sy, sz, x + xi, y + yi, z + zi);

				if (is_26_connected(center, candidate, xi, yi, zi)) {
					subtotal += calc_area_at_point_2x2x2(
						candidate,
						sx, sy, sz,
						delta, pos, normal, anisotropy,
						pts, 
						projections, inv_projections
					);
				}
			}
		}
	}

	return subtotal;
}

template <typename LABEL>
std::tuple<float, uint8_t> cross_sectional_area_helper_2x2x2(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& pos, // plane position
	const Vec3& normal, // plane normal vector
	const Vec3& anisotropy
) {
	const uint64_t grid_size = std::max(((sx+1)>>1) * ((sy+1)>>1) * ((sz+1)>>1), static_cast<uint64_t>(1));
	std::vector<bool> ccl(grid_size);

	uint8_t contact = 0;

	// rational approximation of sqrt(3) is 97/56
	// more reliable behavior across compilers/architectures
	uint64_t plane_size = 2 * 97 * std::max(std::max(sx,sy), sz) / 56 + 1;

	// maximum possible size of plane
	uint64_t psx = plane_size;
	uint64_t psy = psx;

	std::vector<bool> visited(psx * psy);

	Vec3 basis1 = normal.cross(ihat);
	if (basis1.is_null()) {
		basis1 = normal.cross(jhat);
	}
	basis1 /= basis1.norm();

	Vec3 basis2 = normal.cross(basis1);
	basis2 /= basis2.norm();

	uint64_t plane_pos_x = plane_size / 2;
	uint64_t plane_pos_y = plane_size / 2;

	uint64_t ploc = plane_pos_x + psx * plane_pos_y;

	std::stack<uint64_t> stack;
	stack.push(ploc);

	float total = 0.0;

	std::vector<Vec3> pts;
	pts.reserve(6);

	const std::vector<float> projections = {
		ihat.dot(normal),
		jhat.dot(normal),
		khat.dot(normal)
	};

	std::vector<float> inv_projections(3);
	for (int i = 0; i < 3; i++) {
		inv_projections[i] = (projections[i] == 0)
			? 0
			: 1.0 / projections[i];
	}

	const float sxf = static_cast<float>(sx) - 0.5;
	const float syf = static_cast<float>(sy) - 0.5;
	const float szf = static_cast<float>(sz) - 0.5;

	while (!stack.empty()) {
		ploc = stack.top();
		stack.pop();

		if (visited[ploc]) {
			continue;
		}

		visited[ploc] = true;

		uint64_t y = ploc / psx;
		uint64_t x = ploc - y * psx;

		float dx = static_cast<float>(x) - static_cast<float>(plane_pos_x);
		float dy = static_cast<float>(y) - static_cast<float>(plane_pos_y);

		Vec3 cur = pos + basis1 * dx + basis2 * dy;

		if (cur.x <= -0.5 || cur.y <= -0.5 || cur.z <= -0.5) {
			continue;
		}
		else if (cur.x >= sxf || cur.y >= syf || cur.z >= szf) {
			continue;
		}

		cur = cur.round();

		uint64_t loc = (
			static_cast<uint64_t>(cur.x)
			+ sx * (
				static_cast<uint64_t>(cur.y)
				+ sy * static_cast<uint64_t>(cur.z)
			)
		);

		if (labels[loc] != segid) {
			continue;
		}

		contact |= (cur.x < 1); // -x
		contact |= (cur.x >= sx - 1.5) << 1; // +x
		contact |= (cur.y < 1) << 2; // -y
		contact |= (cur.y >= sy - 1.5) << 3; // +y
		contact |= (cur.z < 1) << 4; // -z
		contact |= (cur.z >= sz - 1.5) << 5; // +z

		uint64_t up = ploc - psx; 
		uint64_t down = ploc + psx;
		uint64_t left = ploc - 1;
		uint64_t right = ploc + 1;

		uint64_t upleft = ploc - psx - 1; 
		uint64_t downleft = ploc + psx - 1;
		uint64_t upright = ploc - psx + 1;
		uint64_t downright = ploc + psx + 1;

		if (x > 0 && !visited[left]) {
			stack.push(left);
		}
		if (x < psx - 1 && !visited[right]) {
			stack.push(right);
		}
		if (y > 0 && !visited[up]) {
			stack.push(up);
		}
		if (y < psy - 1 && !visited[down]) {
			stack.push(down);
		}

		if (x > 0 && y > 0 && !visited[upleft]) {
			stack.push(upleft);
		}
		if (x < psx - 1 && y > 0 && !visited[upright]) {
			stack.push(upright);
		}
		if (x > 0 && y < psy - 1 && !visited[downleft]) {
			stack.push(downleft);
		}
		if (x < psx - 1 && y < psy - 1 && !visited[downright]) {
			stack.push(downright);
		}

		total += robust_calc_area_at_point_2x2x2<LABEL>(
			labels, segid, ccl,
			sx, sy, sz,
			cur, pos, normal, anisotropy,
			pts, projections, inv_projections
		);
	}

	return std::make_tuple(total, contact);
}

template <typename LABEL>
std::tuple<float, uint8_t> cross_sectional_area_helper_2x2x2_persistent_data(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& pos, // plane position
	const Vec3& normal, // plane normal vector
	const Vec3& anisotropy
) {
	persistent_data.maybe_grow(sx, sy, sz);
	persistent_data.next_color();

	uint8_t contact = 0;

	// rational approximation of sqrt(3) is 97/56
	// more reliable behavior across compilers/architectures
	uint64_t plane_size = 2 * 97 * std::max(std::max(sx,sy), sz) / 56 + 1;

	// maximum possible size of plane
	uint64_t psx = plane_size;
	uint64_t psy = psx;

	std::vector<bool> visited(psx * psy);

	Vec3 basis1 = normal.cross(ihat);
	if (basis1.is_null()) {
		basis1 = normal.cross(jhat);
	}
	basis1 /= basis1.norm();

	Vec3 basis2 = normal.cross(basis1);
	basis2 /= basis2.norm();

	uint64_t plane_pos_x = plane_size / 2;
	uint64_t plane_pos_y = plane_size / 2;

	uint64_t ploc = plane_pos_x + psx * plane_pos_y;

	std::stack<uint64_t> stack;
	stack.push(ploc);

	float total = 0.0;

	std::vector<Vec3> pts;
	pts.reserve(6);

	const std::vector<float> projections = {
		ihat.dot(normal),
		jhat.dot(normal),
		khat.dot(normal)
	};

	std::vector<float> inv_projections(3);
	for (int i = 0; i < 3; i++) {
		inv_projections[i] = (projections[i] == 0)
			? 0
			: 1.0 / projections[i];
	}

	const float sxf = static_cast<float>(sx) - 0.5;
	const float syf = static_cast<float>(sy) - 0.5;
	const float szf = static_cast<float>(sz) - 0.5;

	while (!stack.empty()) {
		ploc = stack.top();
		stack.pop();

		if (visited[ploc]) {
			continue;
		}

		visited[ploc] = true;

		uint64_t y = ploc / psx;
		uint64_t x = ploc - y * psx;

		float dx = static_cast<float>(x) - static_cast<float>(plane_pos_x);
		float dy = static_cast<float>(y) - static_cast<float>(plane_pos_y);

		Vec3 cur = pos + basis1 * dx + basis2 * dy;

		if (cur.x <= -0.5 || cur.y <= -0.5 || cur.z <= -0.5) {
			continue;
		}
		else if (cur.x >= sxf || cur.y >= syf || cur.z >= szf) {
			continue;
		}

		cur = cur.round();

		uint64_t loc = (
			static_cast<uint64_t>(cur.x)
			+ sx * (
				static_cast<uint64_t>(cur.y)
				+ sy * static_cast<uint64_t>(cur.z)
			)
		);

		if (labels[loc] != segid) {
			continue;
		}

		contact |= (cur.x < 1); // -x
		contact |= (cur.x >= sx - 1.5) << 1; // +x
		contact |= (cur.y < 1) << 2; // -y
		contact |= (cur.y >= sy - 1.5) << 3; // +y
		contact |= (cur.z < 1) << 4; // -z
		contact |= (cur.z >= sz - 1.5) << 5; // +z

		uint64_t up = ploc - psx; 
		uint64_t down = ploc + psx;
		uint64_t left = ploc - 1;
		uint64_t right = ploc + 1;

		uint64_t upleft = ploc - psx - 1; 
		uint64_t downleft = ploc + psx - 1;
		uint64_t upright = ploc - psx + 1;
		uint64_t downright = ploc + psx + 1;

		if (x > 0 && !visited[left]) {
			stack.push(left);
		}
		if (x < psx - 1 && !visited[right]) {
			stack.push(right);
		}
		if (y > 0 && !visited[up]) {
			stack.push(up);
		}
		if (y < psy - 1 && !visited[down]) {
			stack.push(down);
		}

		if (x > 0 && y > 0 && !visited[upleft]) {
			stack.push(upleft);
		}
		if (x < psx - 1 && y > 0 && !visited[upright]) {
			stack.push(upright);
		}
		if (x > 0 && y < psy - 1 && !visited[downleft]) {
			stack.push(downleft);
		}
		if (x < psx - 1 && y < psy - 1 && !visited[downright]) {
			stack.push(downright);
		}

		total += robust_calc_area_at_point_2x2x2_persistent_data(
			labels, segid,
			sx, sy, sz,
			cur, pos, normal, anisotropy,
			pts, projections, inv_projections
		);
	}

	return std::make_tuple(total, contact);
}

template <typename LABEL>
float cross_sectional_area_helper(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& pos, // plane position
	const Vec3& normal, // plane normal vector
	const Vec3& anisotropy, // anisotropy
	uint8_t& contact, 
	float* plane_visualization
) {
	std::vector<bool> ccl(sx * sy * sz);

	// rational approximation of sqrt(3) is 97/56
	// more reliable behavior across compilers/architectures
	uint64_t plane_size = 2 * 97 * std::max(std::max(sx,sy), sz) / 56 + 1;

	// maximum possible size of plane
	uint64_t psx = plane_size;
	uint64_t psy = psx;

	std::vector<bool> visited(psx * psy);

	Vec3 basis1 = normal.cross(ihat);
	if (basis1.is_null()) {
		basis1 = normal.cross(jhat);
	}
	basis1 /= basis1.norm();

	Vec3 basis2 = normal.cross(basis1);
	basis2 /= basis2.norm();

	uint64_t plane_pos_x = plane_size / 2;
	uint64_t plane_pos_y = plane_size / 2;

	uint64_t ploc = plane_pos_x + psx * plane_pos_y;

	std::stack<uint64_t> stack;
	stack.push(ploc);

	float total = 0.0;

	std::vector<Vec3> pts;
	pts.reserve(6);

	const std::vector<float> projections = {
		ihat.dot(normal),
		jhat.dot(normal),
		khat.dot(normal)
	};

	std::vector<float> inv_projections(3);
	for (int i = 0; i < 3; i++) {
		inv_projections[i] = (projections[i] == 0)
			? 0
			: 1.0 / projections[i];
	}

	const float sxf = static_cast<float>(sx) - 0.5;
	const float syf = static_cast<float>(sy) - 0.5;
	const float szf = static_cast<float>(sz) - 0.5;

	while (!stack.empty()) {
		ploc = stack.top();
		stack.pop();

		if (visited[ploc]) {
			continue;
		}

		visited[ploc] = true;

		uint64_t y = ploc / psx;
		uint64_t x = ploc - y * psx;

		float dx = static_cast<float>(x) - static_cast<float>(plane_pos_x);
		float dy = static_cast<float>(y) - static_cast<float>(plane_pos_y);

		Vec3 cur = pos + basis1 * dx + basis2 * dy;

		if (cur.x <= -0.5 || cur.y <= -0.5 || cur.z <= -0.5) {
			continue;
		}
		else if (cur.x >= sxf || cur.y >= syf || cur.z >= szf) {
			continue;
		}

		const uint64_t loc = (
			static_cast<uint64_t>(std::round(cur.x)) 
			+ sx * (
				static_cast<uint64_t>(std::round(cur.y)) 
				+ sy * static_cast<uint64_t>(std::round(cur.z))
			)
		);

		if (labels[loc] != segid) {
			continue;
		}

		contact |= (cur.x < 0.5); // -x
		contact |= (cur.x >= sx - 1.5) << 1; // +x
		contact |= (cur.y < 0.5) << 2; // -y
		contact |= (cur.y >= sy - 1.5) << 3; // +y
		contact |= (cur.z < 0.5) << 4; // -z
		contact |= (cur.z >= sz - 1.5) << 5; // +z

		uint64_t up = ploc - psx; 
		uint64_t down = ploc + psx;
		uint64_t left = ploc - 1;
		uint64_t right = ploc + 1;

		uint64_t upleft = ploc - psx - 1; 
		uint64_t downleft = ploc + psx - 1;
		uint64_t upright = ploc - psx + 1;
		uint64_t downright = ploc + psx + 1;

		if (x > 0 && !visited[left]) {
			stack.push(left);
		}
		if (x < psx - 1 && !visited[right]) {
			stack.push(right);
		}
		if (y > 0 && !visited[up]) {
			stack.push(up);
		}
		if (y < psy - 1 && !visited[down]) {
			stack.push(down);
		}

		if (x > 0 && y > 0 && !visited[upleft]) {
			stack.push(upleft);
		}
		if (x < psx - 1 && y > 0 && !visited[upright]) {
			stack.push(upright);
		}
		if (x > 0 && y < psy - 1 && !visited[downleft]) {
			stack.push(downleft);
		}
		if (x < psx - 1 && y < psy - 1 && !visited[downright]) {
			stack.push(downright);
		}

		total += calc_area_at_point(
			labels, segid, ccl,
			sx, sy, sz,
			cur, pos, normal, anisotropy,
			pts, projections, inv_projections,
			plane_visualization
		);
	}

	return total;
}

};

namespace xs3d {

struct Bbox2d {
	int64_t x_min, x_max;
	int64_t y_min, y_max;
	Bbox2d() : x_min(0), x_max(0), y_min(0), y_max(0) {};
	Bbox2d(int64_t x_min, int64_t x_max, int64_t y_min, int64_t y_max) 
		: x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max) {};

	int64_t sx() const {
		return x_max - x_min;
	}
	int64_t sy() const {
		return y_max - y_min;
	}
	int64_t pixels() const {
		return sx() * sy();
	}
	void print() const {
		printf("Bbox2d(%lld, %lld, %lld, %lld)\n", x_min, x_max, y_min, y_max);
	}
};

void set_shape(
	const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
	persistent_data.init(sx, sy, sz);
}

void clear_shape() {
	persistent_data.clear();
}

template <typename LABEL>
std::tuple<float, uint8_t> cross_sectional_area(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	const bool use_persistent_data = false
) {

	if (px < 0 || px >= sx) {
		return std::make_tuple(0.0, 0);
	}
	else if (py < 0 || py >= sy) {
		return std::make_tuple(0.0, 0);
	}
	else if (pz < 0 || pz >= sz) {
		return std::make_tuple(0.0, 0);
	}

	const Vec3 pos(px, py, pz);
	const Vec3 rpos = pos.round();

	if (
		   rpos.x < 0 || rpos.x >= sx 
		|| rpos.y < 0 || rpos.y >= sy 
		|| rpos.z < 0 || rpos.z >= sz
	) {
		return std::make_tuple(0.0, 0);
	}

	uint64_t loc = static_cast<uint64_t>(rpos.x) + sx * (
		static_cast<uint64_t>(rpos.y) + sy * static_cast<uint64_t>(rpos.z)
	);

	if (loc < 0 || loc >= sx * sy * sz) {
		return std::make_tuple(0.0, 0);
	}
	else if (labels[loc] != segid) {
		return std::make_tuple(0.0, 0);
	}

	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	if (use_persistent_data) {
		return cross_sectional_area_helper_2x2x2_persistent_data<LABEL>(
			labels, segid,
			sx, sy, sz, 
			pos, normal, anisotropy
		);
	}
	else {
		return cross_sectional_area_helper_2x2x2<LABEL>(
			labels, segid,
			sx, sy, sz, 
			pos, normal, anisotropy
		);
	}
}

std::tuple<float*, uint8_t> cross_section(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	float* plane_visualization = NULL
) {
	if (plane_visualization == NULL) {
		plane_visualization = new float[sx * sy * sz]();
	}

	uint8_t contact = 0;

	if (px < 0 || px >= sx) {
		return std::make_tuple(plane_visualization, contact);
	}
	else if (py < 0 || py >= sy) {
		return std::make_tuple(plane_visualization, contact);
	}
	else if (pz < 0 || pz >= sz) {
		return std::make_tuple(plane_visualization, contact);
	}

	const Vec3 pos(px, py, pz);
	const Vec3 rpos = pos.round();

	if (
		   rpos.x < 0 || rpos.x >= sx 
		|| rpos.y < 0 || rpos.y >= sy 
		|| rpos.z < 0 || rpos.z >= sz
	) {
		return std::make_tuple(plane_visualization, contact);
	}

	uint64_t loc = static_cast<uint64_t>(rpos.x) + sx * (
		static_cast<uint64_t>(rpos.y) + sy * static_cast<uint64_t>(rpos.z)
	);

	if (loc < 0 || loc >= sx * sy * sz) {
		return std::make_tuple(plane_visualization, contact);
	}
	else if (!binimg[loc]) {
		return std::make_tuple(plane_visualization, contact);
	}
	
	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	cross_sectional_area_helper(
		binimg, /*segid=*/static_cast<uint8_t>(1),
		sx, sy, sz, 
		pos, normal, anisotropy,
		contact, plane_visualization
	);

	return std::make_tuple(plane_visualization, contact);
}

std::tuple<Vec3, Vec3> create_orthonormal_basis(
	const Vec3& normal, const bool positive_basis
) {
	Vec3 basis1 = normal.cross(jhat);
	if (basis1.is_null()) {
		basis1 = normal.cross(ihat);
	}
	basis1 /= basis1.norm();

	Vec3 basis2 = normal.cross(basis1);
	basis2 /= basis2.norm();

	// try to sort and reflect the bases to approximate
	// a standard basis. First, make basis1 more like the
	// earlier letter of XY, XZ, or YZ and if its
	// pointed into the negatives, reflect it into
	// the positive direction.

	int argmax1 = basis1.abs().argmax();
	int argmax2 = basis2.abs().argmax();

	if (argmax2 < argmax1) {
		std::swap(basis1, basis2);
	}

	Vec3 positive_direction = Vec3(1,1,1);

	if (positive_basis) {
		Vec3 zone = positive_direction;
		if (normal.dot(positive_direction) < 0) {
			zone = -positive_direction;
		}

		if (basis1.dot(zone) < 0) {
			basis1 = -basis1;
		}

		if (basis2.dot(zone) < 0) {
			basis2 = -basis2;
		}
	}

	return std::tuple(basis1, basis2);
}

template <typename LABEL>
std::tuple<LABEL*, Bbox2d> cross_section_projection(
	const LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool c_order,

	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	const bool positive_basis,
	const float crop_distance = std::numeric_limits<float>::infinity(),
	LABEL* out = NULL
) {

	Vec3 anisotropy(wx, wy, wz);

	const float crop_distance_sq = crop_distance * crop_distance / anisotropy.min() / anisotropy.min();

	anisotropy /= anisotropy.min();

	const uint64_t distortion = static_cast<uint64_t>(ceil(
		anisotropy.abs().max()
	));
	anisotropy = Vec3(1,1,1) / anisotropy;

	// maximum possible size of plane
	// rational approximation of sqrt(3) is 97/56
	uint64_t largest_dimension = std::max(std::max(sx,sy), sz);
	if (static_cast<float>(largest_dimension) > crop_distance && crop_distance >= 0) {
		largest_dimension = static_cast<uint64_t>(std::ceil(crop_distance));
	}

	const uint64_t psx = (distortion * 2 * 97 * largest_dimension / 56) + 1;
	const uint64_t psy = psx;

	Bbox2d bbx;

	std::vector<bool> visited(psx * psy);

	if (out == NULL) {
		out = new LABEL[psx * psy]();
	}

	if (px < 0 || px >= sx) {
		return std::tuple(out, bbx);
	}
	else if (py < 0 || py >= sy) {
		return std::tuple(out, bbx);
	}
	else if (pz < 0 || pz >= sz) {
		return std::tuple(out, bbx);
	}

	const Vec3 pos(px, py, pz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	auto bases = create_orthonormal_basis(normal, positive_basis);
	Vec3 basis1 = std::get<0>(bases) * anisotropy;
	Vec3 basis2 = std::get<1>(bases) * anisotropy;

	uint64_t plane_pos_x = psx / 2;
	uint64_t plane_pos_y = psy / 2;

	bbx.x_min = plane_pos_x;
	bbx.x_max = plane_pos_x;
	bbx.y_min = plane_pos_y;
	bbx.y_max = plane_pos_y;

	uint64_t ploc = plane_pos_x + psx * plane_pos_y;

	std::stack<uint64_t> stack;

	if (crop_distance > 0) {
		stack.push(ploc);
	}

	const float sxf = static_cast<float>(sx) - 0.5;
	const float syf = static_cast<float>(sy) - 0.5;
	const float szf = static_cast<float>(sz) - 0.5;

	while (!stack.empty()) {
		ploc = stack.top();
		stack.pop();

		if (visited[ploc]) {
			continue;
		}

		visited[ploc] = true;

		uint64_t y = ploc / psx;
		uint64_t x = ploc - y * psx;

		float dx = static_cast<float>(x) - static_cast<float>(plane_pos_x);
		float dy = static_cast<float>(y) - static_cast<float>(plane_pos_y);

		Vec3 delta = basis1 * dx + basis2 * dy;
		Vec3 cur = pos + delta;

		if (cur.x <= -0.5 || cur.y <= -0.5 || cur.z <= -0.5) {
			continue;
		}
		else if (cur.x >= sxf || cur.y >= syf || cur.z >= szf) {
			continue;
		}
		else if (delta.norm2() >= crop_distance_sq) {
			continue;
		}

		cur = cur.round();

		bbx.x_min = std::min(bbx.x_min, static_cast<int64_t>(x));
		bbx.x_max = std::max(bbx.x_max, static_cast<int64_t>(x));
		bbx.y_min = std::min(bbx.y_min, static_cast<int64_t>(y));
		bbx.y_max = std::max(bbx.y_max, static_cast<int64_t>(y));

		uint64_t loc;
		if (c_order) {
			loc = static_cast<uint64_t>(cur.z) + sz * (
				static_cast<uint64_t>(cur.y) + sy * static_cast<uint64_t>(cur.x)
			);			
		}
		else {
			loc = static_cast<uint64_t>(cur.x) + sx * (
				static_cast<uint64_t>(cur.y) + sy * static_cast<uint64_t>(cur.z)
			);
		}
		 
		out[ploc] = labels[loc];

		uint64_t up = ploc - psx; 
		uint64_t down = ploc + psx;
		uint64_t left = ploc - 1;
		uint64_t right = ploc + 1;

		if (x > 0 && !visited[left]) {
			stack.push(left);
		}
		if (x < psx - 1 && !visited[right]) {
			stack.push(right);
		}
		if (y > 0 && !visited[up]) {
			stack.push(up);
		}
		if (y < psy - 1 && !visited[down]) {
			stack.push(down);
		}
	}

	return std::tuple(out, bbx);
}

};

#include "auxiliary.hpp"

#endif

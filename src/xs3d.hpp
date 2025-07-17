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

using namespace xs3d;

namespace {

static uint8_t _dummy_contact = false;

const Vec3 ihat = Vec3(1,0,0);
const Vec3 jhat = Vec3(0,1,0);
const Vec3 khat = Vec3(0,0,1);

const Vec3 c[8] = {
	Vec3(0, 0, 0), // 0
	Vec3(0, 0, 1), // 1 
	Vec3(0, 1, 0), // 2 
	Vec3(0, 1, 1), // 3
	Vec3(1, 0, 0), // 4
	Vec3(1, 0, 1), // 5
	Vec3(1, 1, 0), // 6
	Vec3(1, 1, 1) // 7
};

const Vec3 pipes[3] = {
	ihat, jhat, khat
};

const Vec3 pipe_points[4] = {
	c[0], c[3], c[5], c[6]
};

void check_intersections(
	std::vector<Vec3>& pts,
	const uint64_t x, const uint64_t y, const uint64_t z,
	const Vec3& pos, const Vec3& normal, 
	const std::vector<float>& projections, 
	const std::vector<float>& inv_projections
) {
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

float calc_area_at_point(
	const uint8_t* binimg,
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

				uint64_t loc = static_cast<uint64_t>(delta.x) + sx * (
					static_cast<uint64_t>(delta.y) + sy * static_cast<uint64_t>(delta.z)
				);

				if (loc < 0 || loc >= voxels) {
					continue;
				}
				else if (!binimg[loc]) {
					continue;
				}

				if (ccl[loc] == 0) {
					ccl[loc] = 1;
					
					check_intersections(
						pts, 
						static_cast<uint64_t>(delta.x), 
						static_cast<uint64_t>(delta.y), 
						static_cast<uint64_t>(delta.z),
						pos, normal, 
						projections, inv_projections
					);

					float area = xs3d::area::points_to_area(pts, anisotropy, normal);
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

float cross_sectional_area_helper(
	const uint8_t* binimg,
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

		if (cur.x < -0.5 || cur.y < -0.5 || cur.z < -0.5) {
			continue;
		}
		else if (cur.x >= (sx - 0.5) || cur.y >= (sy - 0.5) || cur.z >= (sz - 0.5)) {
			continue;
		}

		const uint64_t loc = (
			static_cast<uint64_t>(std::round(cur.x)) 
			+ sx * (
				static_cast<uint64_t>(std::round(cur.y)) 
				+ sy * static_cast<uint64_t>(std::round(cur.z))
			)
		);

		if (!binimg[loc]) {
			continue;
		}

		contact |= (cur.x < 0.5); // -x
		contact |= (cur.x >= sx - 0.5) << 1; // +x
		contact |= (cur.y < 0.5) << 2; // -y
		contact |= (cur.y >= sy - 0.5) << 3; // +y
		contact |= (cur.z < 0.5) << 4; // -z
		contact |= (cur.z >= sz - 0.5) << 5; // +z

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
			binimg, ccl,
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

float cross_sectional_area(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	uint8_t &contact = _dummy_contact 
) {

	if (px < 0 || px >= sx) {
		return 0.0;
	}
	else if (py < 0 || py >= sy) {
		return 0.0;
	}
	else if (pz < 0 || pz >= sz) {
		return 0.0;
	}

	uint64_t loc = static_cast<uint64_t>(px) + sx * (
		static_cast<uint64_t>(py) + sy * static_cast<uint64_t>(pz)
	);

	if (!binimg[loc]) {
		return 0.0;
	}

	const Vec3 pos(px, py, pz);
	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	return cross_sectional_area_helper(
		binimg, 
		sx, sy, sz, 
		pos, normal, anisotropy,
		contact, /*plane_visualization=*/NULL
	);
}

/* This is a version of the cross sectional area calculation
 * that checks every single voxel to ensure that all intersected
 * voxels are included. This is primarily intended for use in
 * testing the standard faster version for correctness.
 *
 * Note that this version does not restrict itself to a single
 * connected component, so pre-filtering must be performed to 
 * ensure a match.
 */
float cross_sectional_area_slow(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	uint8_t &contact = _dummy_contact
) {

	const Vec3 pos(px, py, pz);
	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	std::vector<Vec3> pts;
	pts.reserve(6);

	float area = 0;

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

	contact = 0;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (!binimg[loc]) {
					continue;
				}

				contact |= (x < 1); // -x
				contact |= (x >= sx - 1) << 1; // +x
				contact |= (y < 1) << 2; // -y
				contact |= (y >= sy - 1) << 3; // +y
				contact |= (z < 1) << 4; // -z
				contact |= (z >= sz - 1) << 5; // +z

				check_intersections(
					pts, 
					x, y, z, 
					pos, normal, 
					projections, inv_projections
				);

				area += xs3d::area::points_to_area(pts, anisotropy, normal);
			}
		}
	}

	return area;
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

	uint64_t loc = static_cast<uint64_t>(px) + sx * (
		static_cast<uint64_t>(py) + sy * static_cast<uint64_t>(pz)
	);

	if (!binimg[loc]) {
		return std::make_tuple(plane_visualization, contact);
	}

	const Vec3 pos(px, py, pz);
	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();


	cross_sectional_area_helper(
		binimg, 
		sx, sy, sz, 
		pos, normal, anisotropy,
		contact, plane_visualization
	);

	return std::make_tuple(plane_visualization, contact);
}

std::tuple<float*, uint8_t> cross_section_slow(
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
	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	std::vector<Vec3> pts;
	pts.reserve(6);

	float area = 0;

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

	contact = 0;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (!binimg[loc]) {
					continue;
				}

				contact |= (x < 1); // -x
				contact |= (x >= sx - 1) << 1; // +x
				contact |= (y < 1) << 2; // -y
				contact |= (y >= sy - 1) << 3; // +y
				contact |= (z < 1) << 4; // -z
				contact |= (z >= sz - 1) << 5; // +z

				check_intersections(
					pts, 
					x, y, z, 
					pos, normal, 
					projections, inv_projections
				);

				plane_visualization[loc] = xs3d::area::points_to_area(pts, anisotropy, normal);
			}
		}
	}


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

		if (cur.x < 0 || cur.y < 0 || cur.z < 0) {
			continue;
		}
		else if (cur.x >= sx || cur.y >= sy || cur.z >= sz) {
			continue;
		}
		else if (delta.norm2() >= crop_distance_sq) {
			continue;
		}

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

#endif

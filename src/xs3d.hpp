#ifndef __XS3D_HPP__
#define __XS3D_HPP__

#include "cc3d.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace xs3d {

class Vec3 {
public:
	float x, y, z;
	Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	Vec3 operator+(const Vec3& other) const {
		return Vec3(x + other.x, y + other.y, z + other.z);
	}
	void operator+=(const Vec3& other) {
		x += other.x;
		y += other.y;
		z += other.z;
	}
	Vec3 operator+(const float other) const {
		return Vec3(x + other, y + other, z + other);
	}
	void operator+=(const float other) {
		x += other;
		y += other;
		z += other;
	}
	Vec3 operator-() const {
		return Vec3(-x,-y,-z);
	}
	Vec3 operator-(const Vec3& other) const {
		return Vec3(x - other.x, y - other.y, z - other.z);
	}
	Vec3 operator*(const float scalar) const {
		return Vec3(x * scalar, y * scalar, z * scalar);
	}
	void operator*=(const float scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	Vec3 operator*(const Vec3& other) const {
		return Vec3(x * other.x, y * other.y, z * other.z);
	}
	void operator*=(const Vec3& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
	}
	Vec3 operator/(const float divisor) const {
		return Vec3(x/divisor, y/divisor, z/divisor);
	}
	void operator/=(const float divisor) {
		x /= divisor;
		y /= divisor;
		z /= divisor;
	}
	bool operator==(const Vec3& other) const {
		return x == other.x && y == other.y && z == other.z;
	}

	float dot(const Vec3& o) const {
		return x * o.x + y * o.y + z * o.z;
	}
	float norm() const {
		return sqrt(x*x + y*y + z*z);
	}
	bool close(const Vec3& o) {
		return (*this - o).norm() < 1e-4;
	}
	Vec3 cross(const Vec3& o) const {
		return Vec3(
			y * o.z - z * o.y, 
			z * o.x - x * o.z,
			x * o.y - y * o.x
		);
	}
};

uint32_t* compute_ccl(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const float px, const float py, const float pz, // plane position
	const float nx, const float ny, const float nz 	// plane normal vector
) {

	uint8_t* markup = new uint8_t[sx*sy*sz]();

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (!binimg[loc]) {
					continue;
				}

				float fx = static_cast<float>(x);
				float fy = static_cast<float>(y);
				float fz = static_cast<float>(z);

				// cp_x = current to plane x element
				float cp_x = fx - px;
				float cp_y = fy - py;
				float cp_z = fz - pz;

				float dot_product = cp_x * nx + cp_y * ny + cp_z * nz;
				float proj_x = dot_product * nx;
				float to_plane_x = cp_x - proj_x;

				// pt is the point on the plane
				float pt_x = px + to_plane_x;

				// Why 0.505 instead of 0.5? There is a subtle
				// geometry where the plane is slightly inclined
				// but the closet point to the center is at just over 
				// 0.5, but a substantial part of the plane still
				// intersects the cube.
				const float bounds = 0.505;
				if (pt_x < fx - bounds || pt_x > fx + bounds) {
					continue;
				}

				float proj_y = dot_product * ny;
				float to_plane_y = cp_y - proj_y;
				float pt_y = py + to_plane_y;

				if (pt_y < fy - bounds || pt_y > fy + bounds) {
					continue;
				}

				float proj_z = dot_product * nz;
				float to_plane_z = cp_z - proj_z;
				float pt_z = pz + to_plane_z;

				if (pt_z < fz - bounds || pt_z > fz + bounds) {
					continue;
				}

				const float eps = 0.5001;
				const uint64_t zero = 0;
				uint64_t ipt_x = std::max(std::min(static_cast<uint64_t>(pt_x + eps), sx - 1), zero);
				uint64_t ipt_y = std::max(std::min(static_cast<uint64_t>(pt_y + eps), sy - 1), zero);
				uint64_t ipt_z = std::max(std::min(static_cast<uint64_t>(pt_z + eps), sz - 1), zero);

				uint64_t pt_loc = ipt_x + sx * (ipt_y + sy * ipt_z);
				markup[pt_loc] = static_cast<uint8_t>(binimg[pt_loc]);
			}
		}
	}

	uint32_t* ccl = cc3d::connected_components3d<uint8_t, uint32_t>(
		markup, sx, sy, sz
	);
	delete[] markup;

	return ccl;
}

float area_of_triangle(
	const std::vector<Vec3>& pts, 
	const Vec3& anisotropy
) {
	Vec3 v1 = pts[1] - pts[0];
	v1 *= anisotropy;
	Vec3 v2 = pts[1] - pts[0];
	v2 *= anisotropy;
	Vec3 v3 = v1.cross(v2);
	return v3.norm() / 2.0;
}

float area_of_quad(
	const std::vector<Vec3>& pts, 
	const Vec3& anisotropy
) {
	Vec3 v1 = pts[1] - pts[0];
	v1 *= anisotropy;
	Vec3 v2 = pts[2] - pts[0];
	v2 *= anisotropy;
	Vec3 v3 = pts[3] - pts[0];
	v3 *= anisotropy;

	float norm1 = v1.norm();
	float norm2 = v2.norm();
	float norm3 = v3.norm();

  // remove the most distant point so we are
  // not creating a faulty quad based on the 
  // diagonal. Use a decision tree since it's
  // both more efficient and less annoying 
  // than some list operations.

	if (norm1 > norm2) {
		if (norm1 > norm3) {
			return v2.cross(v3).norm();
		}
		else if (norm3 > norm2) {
			return v1.cross(v2).norm();
		}
		else {
			return v1.cross(v3).norm();
		}
	}
	else if (norm2 > norm3) {
		if (norm1 > norm3) {
			return v1.cross(v3).norm();
		}
		else {
			return v1.cross(v2).norm();
		}
	}
	else {
		return v1.cross(v2).norm();
	}
}

float area_of_poly(
	const std::vector<Vec3>& pts, 
	const Vec3& normal,
	const Vec3& anisotropy
) {
	
	Vec3 centroid(0,0,0);

	for (Vec3 pt : pts) {
		centroid += pt;
	}
	centroid /= static_cast<float>(pts.size());

	std::vector<Vec3> spokes;
	for (Vec3 pt : pts) {
		spokes.push_back(pt - centroid);
	}

	Vec3 prime_spoke = (pts[0] - centroid);

	Vec3 basis = prime_spoke.cross(normal);
	basis /= basis.norm();
	
	auto angularOrder = [&](const Vec3& a, const Vec3& b) {
		float cosine = a.dot(prime_spoke) / a.norm();
		float a_angle = std::acos(cosine);

		if (a.dot(basis) < 0) {
			a_angle = -a_angle;
		}

		cosine = b.dot(prime_spoke) / b.norm();
		float b_angle = std::acos(cosine);
		
		if (b.dot(basis) < 0) {
			b_angle = -b_angle;
		}

		return a_angle < b_angle;
	};

	std::sort(spokes.begin(), spokes.end(), angularOrder);

	for (Vec3& spoke : spokes) {
		spoke *= anisotropy;
	}

	float area = 0.0;
	for (int i = 0; i < spokes.size() - 1; i++) {
		area += spokes[i].cross(spokes[i+1]).norm() / 2.0;
	}
	area += spokes[0].cross(spokes[spokes.size() - 1]).norm() / 2.0;

	return area;
}

void check_intersections(
	std::vector<Vec3>& pts,
	const uint64_t x, const uint64_t y, const uint64_t z,
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz
) {
	pts.clear();

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

	const Vec3 ihat = Vec3(1, 0, 0);
	const Vec3 jhat = Vec3(0, 1, 0);
	const Vec3 khat = Vec3(0, 0, 1);

	Vec3 pipes[12] = {
		ihat, ihat, ihat, ihat,
		jhat, jhat, jhat, jhat,
		khat, khat, khat, khat
	};

	Vec3 pipe_points[12] = {
		c[0], c[1], c[2], c[3],
		c[0], c[1], c[4], c[5],
		c[0], c[2], c[4], c[6]
	};

	Vec3 xyz(x,y,z);
	xyz += -0.5;

	Vec3 pos(px,py,pz);
	Vec3 normal(nx,ny,nz);

	auto inlist = [&](const Vec3& pt){
		for (Vec3& p : pts) {
			if (p.close(pt)) {
				return true;
			}
		}
		return false;
	};

	for (int i = 0; i < 12; i++) {
		Vec3 pipe = pipes[i];
		Vec3 corner = pipe_points[i];
		corner += xyz;
		
		Vec3 cur_vec = pos - corner;
		float proj = cur_vec.dot(normal);

		if (proj == 0) {
			if (!inlist(corner)) {
				pts.push_back(corner);
			}
			continue;
		}

		float proj2 = pipe.dot(normal);

		// if traveling parallel to plane but
		// not on the plane
		if (proj2 == 0) {
			continue;
		}

		float t = proj / proj2;
		Vec3 nearest_pt = corner + pipe * t;

		if (nearest_pt.x > (x+0.5) || nearest_pt.x < (x-0.5)) {
			continue;
		}
		else if (nearest_pt.y > (y+0.5) || nearest_pt.y < (y-0.5)) {
			continue;
		}
		else if (nearest_pt.z > (z+0.5) || nearest_pt.z < (z-0.5)) {
			continue;
		}

		if (!inlist(nearest_pt)) {
			pts.push_back(nearest_pt);
		}
	}
}

float cross_sectional_area(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx = 1.0, const float wy = 1.0, const float wz = 1.0
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

	const Vec3 anisotropy(wx, wy, wz);
	Vec3 nhat(nx, ny, nz);
	nhat /= nhat.norm();

	uint32_t* ccl = compute_ccl(
		binimg, 
		sx, sy, sz, 
		px, py, pz, 
		nhat.x, nhat.y, nhat.z
	);

	const uint32_t label = ccl[loc];

	std::vector<Vec3> pts;
	pts.reserve(12);

	float total = 0.0;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			for (uint64_t x = 0; x < sx; x++) {
				loc = x + sx * (y + sy * z);
				if (ccl[loc] != label) {
					continue;
				}

				check_intersections(
					pts, 
					x, y, z,
					px, py, pz,
					nhat.x, nhat.y, nhat.z
				);

				const auto size = pts.size();

				if (size < 3) {
					// no contact, point, or line which have zero area
					continue;
				}
				else if (size > 6) {
					printf("size: %d", size);
					for (auto pt : pts) {
						printf("p %.2f %.2f %.2f\n", pt.x, pt.y, pt.z);
					}
					return -1.0;
				}
				else if (size == 3) {
					total += area_of_triangle(pts, anisotropy);
				}
				else if (size == 4) { 
					total += area_of_quad(pts, anisotropy);
				}
				else { // 5, 6
					total += area_of_poly(pts, nhat, anisotropy);
				}
			}
		}
	}

	delete[] ccl;

	return total;
}

};

#endif
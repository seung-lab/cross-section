#ifndef __XS3D_HPP__
#define __XS3D_HPP__

#include <cmath>
#include <cstdint>
#include <memory>
#include <stack>
#include <vector>

namespace {

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
	bool is_null() const {
		return x == 0 && y == 0 && z == 0;
	}
	bool is_axis_aligned() const {
		return ((x != 0) + (y != 0) + (z != 0)) == 1;
	}
	void print(const std::string &name) {
		printf("%s %.3f, %.3f, %.3f\n",name.c_str(), x, y, z);
	}
};

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
	for (uint64_t i = 0; i < spokes.size() - 1; i++) {
		area += spokes[i].cross(spokes[i+1]).norm() / 2.0;
	}
	area += spokes[0].cross(spokes[spokes.size() - 1]).norm() / 2.0;

	return area;
}

void check_intersections(
	std::vector<Vec3>& pts,
	const uint64_t x, const uint64_t y, const uint64_t z,
	const Vec3& pos, const Vec3& normal
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

float calc_area_at_point(
	const uint8_t* binimg,
	std::vector<bool>& ccl,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const Vec3& cur, const Vec3& pos, 
	const Vec3& normal, const Vec3& anisotropy,
	std::vector<Vec3>& pts
) {
	float subtotal = 0.0;

	float xs = -1;
	float ys = -1;
	float zs = -1;

	float xe = 1;
	float ye = 1;
	float ze = 1;
	
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

				if (delta.x < 0 || delta.y < 0 || delta.z < 0) {
					continue;
				}
				else if (delta.x >= sx || delta.y >= sy || delta.z >= sz) {
					continue;
				}

				uint64_t loc = static_cast<uint64_t>(delta.x) + sx * (
					static_cast<uint64_t>(delta.y) + sy * static_cast<uint64_t>(delta.z)
				);

				if (!binimg[loc]) {
					continue;
				}

				if (ccl[loc] == 0) {
					ccl[loc] = 1;
					
					check_intersections(
						pts, 
						static_cast<uint64_t>(delta.x), 
						static_cast<uint64_t>(delta.y), 
						static_cast<uint64_t>(delta.z),
						pos, normal
					);

					const auto size = pts.size();

					if (size < 3) {
						// no contact, point, or line which have zero area
						continue;
					}
					else if (size > 6) {
						throw new std::runtime_error("Invalid polygon.");
					}
					else if (size == 3) {
						subtotal += area_of_triangle(pts, anisotropy);
					}
					else if (size == 4) { 
						subtotal += area_of_quad(pts, anisotropy);
					}
					else { // 5, 6
						subtotal += area_of_poly(pts, normal, anisotropy);
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
	const Vec3& anisotropy // anisotropy
) {
	std::vector<bool> ccl(sx * sy * sz);

	uint64_t diagonal = static_cast<uint64_t>(ceil(sqrt(sx * sx + sy * sy + sz * sz)));

	// maximum possible size of plane multiplied by sampling frequency
	uint64_t psx = diagonal;
	uint64_t psy = psx;

	std::unique_ptr<bool[]> visited(new bool[psx * psy]());

	Vec3 ihat = Vec3(1,0,0);
	Vec3 jhat = Vec3(0,1,0);

	Vec3 basis1 = normal.cross(ihat);
	if (basis1.is_null()) {
		basis1 = normal.cross(jhat);
	}
	basis1 /= basis1.norm();

	Vec3 basis2 = normal.cross(basis1);
	basis2 /= basis2.norm();

	uint64_t plane_pos_x = diagonal / 2;
	uint64_t plane_pos_y = diagonal / 2;

	uint64_t ploc = plane_pos_x + psx * plane_pos_y;

	std::stack<uint64_t> stack;
	stack.push(ploc);

	float total = 0.0;

	std::vector<Vec3> pts;
	pts.reserve(12);

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

		if (cur.x < 0 || cur.y < 0 || cur.z < 0) {
			continue;
		}
		else if (cur.x >= sx || cur.y >= sy || cur.z >= sz) {
			continue;
		}

		uint64_t loc = static_cast<uint64_t>(cur.x) + sx * (
			static_cast<uint64_t>(cur.y) + sy * static_cast<uint64_t>(cur.z)
		);

		if (!binimg[loc]) {
			continue;
		}

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
			pts
		);
	}

	return total;
}

};

namespace xs3d {

float cross_sectional_area(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz
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
		pos, normal, anisotropy
	);
}

};

#endif

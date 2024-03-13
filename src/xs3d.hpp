#ifndef __XS3D_HPP__
#define __XS3D_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

static uint8_t _dummy_contact = false;

class Vec3 {
public:
	float x, y, z;
	Vec3() : x(0), y(0), z(0) {}
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
	Vec3 operator-(const float scalar) const {
		return Vec3(x - scalar, y - scalar, z - scalar);
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
	Vec3 operator/(const Vec3& other) const {
		return Vec3(x/other.x, y/other.y, z/other.z);
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
	float& operator[](const int idx) {
		if (idx == 0) {
			return x;
		}
		else if (idx == 1) {
			return y;
		}
		else if (idx == 2) {
			return z;
		}
		else {
			throw new std::runtime_error("Index out of bounds.");
		}
	}
	float dot(const Vec3& o) const {
		return x * o.x + y * o.y + z * o.z;
	}
	Vec3 abs() const {
		return Vec3(std::abs(x), std::abs(y), std::abs(z));
	}
	int argmax() const {
		if (x >= y) {
			return (x >= z) ? 0 : 2;
		}
		return (y >= z) ? 1 : 2;
	}
	float max() const {
		return std::max(x,std::max(y,z));
	}
	float min() const {
		return std::min(x,std::min(y,z));
	}
	float norm() const {
		return sqrt(x*x + y*y + z*z);
	}
	float norm2() const {
		return x*x + y*y + z*z;
	}
	bool close(const Vec3& o) const {
		return (*this - o).norm2() < 1e-4;
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
	int num_zero_dims() const {
		return (x == 0) + (y == 0) + (z == 0);
	}
	bool is_axis_aligned() const {
		return ((x != 0) + (y != 0) + (z != 0)) == 1;
	}
	void print(const std::string &name) const {
		printf("%s %.3f, %.3f, %.3f\n",name.c_str(), x, y, z);
	}
};

const Vec3 ihat = Vec3(1,0,0);
const Vec3 jhat = Vec3(0,1,0);
const Vec3 khat = Vec3(0,0,1);

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

// area of a quad is ||v1 x v2||
// but there are two situations:
// in the first, you have the vectors
// representing the length and width
// of the rectangle, which is the usual
// case and the defintion of the cross product
// as a determinant -> area of a paralllepiped 
// works. 
// The second situation is a side and a hypotenuse
// of the rectangle. In that case, let v be a side and
// h be the hypotenuse. ||v x h|| / 2 = area of triangle
// However, the other triangle has the same formula, and
// 2 * || v x h || / 2 = || v x h ||

// so weirdly, you can just pick two vectors 
// and cross them!
float area_of_quad(
	const std::vector<Vec3>& pts, 
	const Vec3& anisotropy
) {
	Vec3 v1 = pts[1] - pts[0];
	v1 *= anisotropy;
	Vec3 v2 = pts[2] - pts[0];
	v2 *= anisotropy;
	
	return v1.cross(v2).norm();
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
	prime_spoke /= prime_spoke.norm();

	Vec3 basis = prime_spoke.cross(normal);
	basis /= basis.norm();

	// acos is expensive, but we just need a number that
	// increments from 0 to a maximum value at 180deg so
	// just do 1 - cos, which will be 0 at 0deg and 2 at 180deg
	auto angularOrder = [&](const Vec3& a, const Vec3& b) {
		float a_val = 1 - (a.dot(prime_spoke) / a.norm());
		if (a.dot(basis) < 0) {
			a_val = -a_val;
		}

		float b_val = 1 - (b.dot(prime_spoke) / b.norm());
		if (b.dot(basis) < 0) {
			b_val = -b_val;
		}

		return a_val < b_val;
	};

	std::sort(spokes.begin(), spokes.end(), angularOrder);

	for (Vec3& spoke : spokes) {
		spoke *= anisotropy;
	}

	float area = 0.0;
	for (uint64_t i = 0; i < spokes.size() - 1; i++) {
		area += spokes[i].cross(spokes[i+1]).norm();
	}
	area += spokes[0].cross(spokes[spokes.size() - 1]).norm();

	return area * 0.5;
}

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
						pos, normal, 
						projections, inv_projections
					);

					const auto size = pts.size();

					float area = 0.0;

					if (size < 3) {
						// no contact, point, or line which have zero area
						continue;
					}
					else if (size > 6) {
						throw new std::runtime_error("Invalid polygon.");
					}
					else if (size == 3) {
						area = area_of_triangle(pts, anisotropy);
					}
					else if (size == 4) { 
						area = area_of_quad(pts, anisotropy);
					}
					else { // 5, 6
						area = area_of_poly(pts, normal, anisotropy);
					}

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

		contact |= (cur.x < 1); // -x
		contact |= (cur.x >= sx - 1) << 1; // +x
		contact |= (cur.y < 1) << 2; // -y
		contact |= (cur.y >= sy - 1) << 3; // +y
		contact |= (cur.z < 1) << 4; // -z
		contact |= (cur.z >= sz - 1) << 5; // +z

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
		printf("Bbox2d(%llu, %llu, %llu, %llu)\n", x_min, x_max, y_min, y_max);
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
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	const bool positive_basis,
	LABEL* out = NULL
) {

	Vec3 anisotropy(wx, wy, wz);
	anisotropy /= anisotropy.min();
	const uint64_t distortion = static_cast<uint64_t>(ceil(
		anisotropy.abs().max()
	));
	anisotropy = Vec3(1,1,1) / anisotropy;

	// maximum possible size of plane
	// rational approximation of sqrt(3) is 97/56
	const uint64_t psx = (distortion * 2 * 97 * std::max(std::max(sx,sy), sz) / 56) + 1;
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
	stack.push(ploc);

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

		bbx.x_min = std::min(bbx.x_min, static_cast<int64_t>(x));
		bbx.x_max = std::max(bbx.x_max, static_cast<int64_t>(x));
		bbx.y_min = std::min(bbx.y_min, static_cast<int64_t>(y));
		bbx.y_max = std::max(bbx.y_max, static_cast<int64_t>(y));

		uint64_t loc = static_cast<uint64_t>(cur.x) + sx * (
			static_cast<uint64_t>(cur.y) + sy * static_cast<uint64_t>(cur.z)
		);

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

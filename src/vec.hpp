#ifndef __XS3D_VEC_HPP__
#define __XS3D_VEC_HPP__

#include <string>

namespace xs3d {

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
	void operator-=(const Vec3& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}
	void operator-=(const float scalar) {
		x -= scalar;
		y -= scalar;
		z -= scalar;
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
	Vec3 round() const {
		return Vec3(std::round(x), std::round(y), std::round(z));
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


};

#endif 

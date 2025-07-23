#include "src/xs3d.hpp"

// g++ -std=c++17 -O2 -g test.cpp -o test

void perf() {
	const int sx = 500;
	const int sy = 500;
	const int sz = 500;

	const int sxy = sx * sy;
	const int voxels = sx * sy * sz;

	uint8_t* binimg = new uint8_t[voxels]();
	std::fill(binimg, binimg + voxels, 1);

	for (int i = 0; i < sx; i++) {
		auto area = xs3d::cross_sectional_area(
			binimg,
			sx, sy, sz,
			/*px=*/i, /*py=*/sy/2, /*pz=*/sz/2,
			/*nx=*/1, /*ny=*/1, /*nz=*/1,
			/*wx=*/1, /*wy=*/1, /*wz=*/1
		);
	}
	
	delete[] binimg;
}

int main() {
	perf();
	return 0;
}
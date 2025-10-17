// xs3d functions mainly intended for debugging

#ifndef __XS3D_AUX_HPP__
#define __XS3D_AUX_HPP__

namespace xs3d {

std::tuple<float*, uint8_t> cross_section_slow_2x2x2(
	const uint8_t* binimg,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz,
	float* plane_visualization = NULL
) {
	const uint64_t grid_size = std::max((sx * sy * sz + 7) >> 3, static_cast<uint64_t>(1));

	if (plane_visualization == NULL) {
		plane_visualization = new float[grid_size]();
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

	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

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

	contact = 0;

	for (uint64_t z = 0; z < sz; z += 2) {
		for (uint64_t y = 0; y < sy; y += 2) {
			for (uint64_t x = 0; x < sx; x += 2) {
				uint64_t loc = x + sx * (y + sy * z);
				uint64_t ploc = (x >> 1) + ((sx+1) >> 1) * ((y >> 1) + ((sy+1) >> 1) * (z >> 1));

				if (!binimg[loc]) {
					continue;
				}

				check_intersections_2x2x2(
					pts, 
					x, y, z, 
					pos, normal, 
					projections, inv_projections
				);

				plane_visualization[ploc] = xs3d::area::points_to_area(pts, anisotropy, normal);
			}
		}
	}

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

	const Vec3 rpos = pos.round();

	if (
		   rpos.x < 0 || rpos.x >= sx 
		|| rpos.y < 0 || rpos.y >= sy 
		|| rpos.z < 0 || rpos.z >= sz
	) {
		return std::make_tuple(plane_visualization, contact);
	}

	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

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

				check_intersections_1x1x1(
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


/* This is a version of the cross sectional area calculation
 * that checks every single voxel to ensure that all intersected
 * voxels are included. This is primarily intended for use in
 * testing the standard faster version for correctness.
 *
 * Note that this version does not restrict itself to a single
 * connected component, so pre-filtering must be performed to 
 * ensure a match.
 */
template <typename LABEL>
std::tuple<float, uint8_t> cross_sectional_area_slow(
	const LABEL* labels, const LABEL segid,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	
	const float px, const float py, const float pz,
	const float nx, const float ny, const float nz,
	const float wx, const float wy, const float wz
) {

	const Vec3 pos(px, py, pz);
	const Vec3 rpos = pos.round();

	if (
		   rpos.x < 0 || rpos.x >= sx 
		|| rpos.y < 0 || rpos.y >= sy 
		|| rpos.z < 0 || rpos.z >= sz
	) {
		return std::make_tuple(0.0, 0);
	}

	const Vec3 anisotropy(wx, wy, wz);
	Vec3 normal(nx, ny, nz);
	normal /= normal.norm();

	std::vector<Vec3> pts;
	pts.reserve(6);

	double area = 0;

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

	uint8_t contact = 0;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (labels[loc] != segid) {
					continue;
				}

				contact |= (x < 1); // -x
				contact |= (x >= sx - 1) << 1; // +x
				contact |= (y < 1) << 2; // -y
				contact |= (y >= sy - 1) << 3; // +y
				contact |= (z < 1) << 4; // -z
				contact |= (z >= sz - 1) << 5; // +z

				check_intersections_1x1x1(
					pts, 
					x, y, z, 
					pos, normal, 
					projections, inv_projections
				);

				area += xs3d::area::points_to_area(pts, anisotropy, normal);
			}
		}
	}

	return std::make_tuple(area, contact);
}

};

#endif
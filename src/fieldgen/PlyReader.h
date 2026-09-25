#pragma once

#include "macrofacet/core/Types.h"
#include <Eigen/Core>
#include <filesystem>
#include <string>

namespace mf {

struct TriangleMesh {
    Eigen::MatrixXd vertices;  // n x 3
    Eigen::MatrixXi faces;     // m x 3

    Bounds3 bounds() const;
};

// Reads the subset of PLY the field generator needs: vertex x/y/z and triangle
// face indices, from either `ascii` or `binary_little_endian`. Every other
// element and property is parsed only far enough to be skipped correctly.
//
// Throws std::runtime_error on malformed input. The error messages name the
// line or byte offset, because a silently-misparsed PLY produces a plausible
// but wrong distance field, which is far more expensive to debug.
TriangleMesh readPlyMesh(const std::filesystem::path& path);

} // namespace mf

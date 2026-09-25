#pragma once

#include "fieldgen/PlyReader.h"

#include <igl/AABB.h>
#include <igl/fast_winding_number.h>

#include <Eigen/Core>

namespace mf {

// How the sign of the distance is decided from the generalized winding number w.
enum class SignMode {
    // The original generator's formula: dist = sqrt(sqrD) * (1 - 2|w|).
    // Exact for closed meshes, where |w| is 1 inside and 0 outside. Note the
    // winding approximation is multiplied into the *magnitude*, not just used
    // for the sign.
    Winding,
    // dist = sign(w < 0.5) * sqrt(sqrD). The magnitude is left untouched, so a
    // non-watertight or self-intersecting mesh (where w can sit near 0.5) cannot
    // collapse the field toward zero.
    Threshold,
};

// Signed distance to a triangle mesh, via libigl's AABB tree for the unsigned
// distance and its fast winding number for the sign.
//
// The 5-argument single-point squared_distance overload is intentional: the
// 6-argument overload takes `up_sqr_d` (an upper bound) rather than an output
// parameter, so passing a squared-distance accumulator there compiles cleanly
// and silently returns nothing. See docs/archive/PLAN_NANOVDB_FIELD.md 6.0.
class MeshDistanceField {
public:
    explicit MeshDistanceField(const TriangleMesh& mesh, SignMode signMode = SignMode::Winding);

    const Bounds3& bounds() const { return bounds_; }

    double unsignedDistance(const Point3& p) const;
    double windingNumber(const Point3& p) const;
    double signedDistance(const Point3& p) const;

    // Batched unsigned distance -- the form the bake loop uses. `sqrD` is
    // resized to points.rows(); `nearestTriangle` may be null.
    void unsignedSquaredDistances(const Eigen::MatrixXd& points, Eigen::VectorXd& sqrD,
                                  Eigen::VectorXi* nearestTriangle = nullptr) const;

    std::size_t triangleCount() const { return static_cast<std::size_t>(faces_.rows()); }

private:
    Eigen::MatrixXd vertices_;
    Eigen::MatrixXi faces_;
    igl::AABB<Eigen::MatrixXd, 3> tree_;
    igl::FastWindingNumberBVH windingBvh_;
    SignMode signMode_ = SignMode::Winding;
    Bounds3 bounds_;
};

} // namespace mf

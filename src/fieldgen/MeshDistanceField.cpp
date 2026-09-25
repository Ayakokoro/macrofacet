#include "fieldgen/MeshDistanceField.h"

#include <cmath>
#include <stdexcept>

namespace mf {

MeshDistanceField::MeshDistanceField(const TriangleMesh& mesh, SignMode signMode)
    : vertices_(mesh.vertices), faces_(mesh.faces), signMode_(signMode) {
    if (vertices_.rows() == 0 || faces_.rows() == 0) {
        throw std::invalid_argument("MeshDistanceField needs a non-empty triangle mesh");
    }
    tree_.init(vertices_, faces_);
    // order 2 is what the original generator used and what the plan measured.
    igl::fast_winding_number(vertices_, faces_, 2, windingBvh_);
    bounds_ = mesh.bounds();
}

// 5 arguments: (V, F, p, i, c) -> Scalar. Deliberately NOT the 6-argument form,
// whose 4th parameter is `up_sqr_d`, an upper bound rather than an output.
double MeshDistanceField::unsignedDistance(const Point3& p) const {
    const Eigen::RowVector3d query = p.transpose();
    int nearest = -1;
    Eigen::RowVector3d closest;
    const double squared = tree_.squared_distance(vertices_, faces_, query, nearest, closest);
    return std::sqrt(std::max(0.0, squared));
}

double MeshDistanceField::windingNumber(const Point3& p) const {
    return igl::fast_winding_number(windingBvh_, 2.0f, p.transpose());
}

double MeshDistanceField::signedDistance(const Point3& p) const {
    const double magnitude = unsignedDistance(p);
    const double winding = windingNumber(p);
    if (signMode_ == SignMode::Threshold) {
        return winding < 0.5 ? magnitude : -magnitude;
    }
    // The original formula. (1 - 2|w|) is +1 outside (w == 0) and -1 inside
    // (|w| == 1) for a closed mesh; in between it scales the magnitude down.
    return magnitude * (1.0 - 2.0 * std::abs(winding));
}

void MeshDistanceField::unsignedSquaredDistances(const Eigen::MatrixXd& points,
                                                 Eigen::VectorXd& sqrD,
                                                 Eigen::VectorXi* nearestTriangle) const {
    sqrD.resize(points.rows());
    Eigen::VectorXi nearest;
    Eigen::MatrixXd closest;
    // The batched overload takes sqrD as an output, which is the one we want.
    tree_.squared_distance(vertices_, faces_, points, sqrD, nearest, closest);
    if (nearestTriangle) *nearestTriangle = std::move(nearest);
}

} // namespace mf

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf::detail {

// For an isotropic trilinear grid, each partial derivative is a convex
// combination of corner differences divided by dx. If all stored and
// background values lie in [minimum, maximum], every component is bounded by
// (maximum - minimum) / dx, so the 3D gradient norm is bounded by sqrt(3)
// times that value. The old single-component bound was insufficient at corners
// where derivatives in several axes are simultaneously large.
inline double trilinearGradientNormBound(double minimum, double maximum, double dx) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
        !std::isfinite(dx) || maximum < minimum || !(dx > 0.0)) {
        throw std::invalid_argument("invalid trilinear gradient bound inputs");
    }
    const double bound = std::sqrt(3.0) * ((maximum - minimum) / dx);
    if (!std::isfinite(bound)) {
        throw std::invalid_argument("trilinear gradient bound is not finite");
    }
    return bound == 0.0 ? 0.0 :
        std::nextafter(bound, std::numeric_limits<double>::infinity());
}

// The gradient inside a trilinear cell is a convex combination of its eight
// one-sided corner gradients. Its maximum norm is therefore attained at a
// corner. Values include the grid background wherever a voxel is inactive.
inline double trilinearCellGradientNormBound(const double (&value)[2][2][2], double dx) {
    if (!(dx > 0.0) || !std::isfinite(dx)) {
        throw std::invalid_argument("invalid trilinear cell width");
    }
    double maximum = 0.0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                if (!std::isfinite(value[i][j][k])) {
                    throw std::invalid_argument("non-finite trilinear cell value");
                }
                const double gx = (value[1][j][k] - value[0][j][k]) / dx;
                const double gy = (value[i][1][k] - value[i][0][k]) / dx;
                const double gz = (value[i][j][1] - value[i][j][0]) / dx;
                const double norm = std::hypot(gx, gy, gz);
                if (!std::isfinite(norm)) {
                    throw std::invalid_argument("trilinear cell gradient bound is not finite");
                }
                maximum = std::max(maximum, norm);
            }
        }
    }
    // Allow for the arithmetic in sampleWithGradient as well as the bound.
    const double inflated = maximum * (1.0 + 64.0 * std::numeric_limits<double>::epsilon());
    if (!std::isfinite(inflated)) {
        throw std::invalid_argument("trilinear cell gradient bound overflowed");
    }
    return maximum == 0.0 ? 0.0 :
        std::nextafter(inflated, std::numeric_limits<double>::infinity());
}

} // namespace mf::detail

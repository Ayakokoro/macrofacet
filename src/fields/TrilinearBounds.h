#pragma once

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

} // namespace mf::detail

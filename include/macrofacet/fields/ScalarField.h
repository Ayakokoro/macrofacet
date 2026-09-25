#pragma once

#include "macrofacet/core/Types.h"
#include <memory>

namespace mf {

// Bounds for an interpolated scalar field sampled at arbitrary points. Density
// and roughness use the same contract for conservative transport bounds.
struct ScalarBounds {
    double minimumValue = 0.0;
    double maximumValue = 0.0;
    bool certified = false;
};

// A per-point scalar grid, used for transport density and material alpha.
// Neither quantity is the GP mean field: density defines the active transport
// band, while alpha is evaluated by the material at collision points.
class ScalarField {
public:
    virtual ~ScalarField() = default;
    virtual double sample(const Point3& x) const = 0;
    // Must be conservative: for every x in domain, sample(x) <= maximumValue.
    virtual ScalarBounds bounds(const Bounds3& domain) const = 0;
};

using ScalarFieldPtr = std::shared_ptr<const ScalarField>;

} // namespace mf

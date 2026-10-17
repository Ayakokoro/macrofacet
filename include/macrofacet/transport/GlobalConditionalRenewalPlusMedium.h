#pragma once

#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

struct GlobalConditionalRenewalPlusSample {
    bool collided = false;
    double age = 0.0; // World distance from the original flight birth.
    Vector3 gradient = Vector3::Zero(); // Full, unnormalized world gradient.
};

// Uses the original global mean and kernel throughout a flight. At fixed world
// distances, replace the birth observation with a jointly sampled positive
// endpoint value and full gradient. Earlier observations are discarded.
class GlobalConditionalRenewalPlusMedium {
public:
    GlobalConditionalRenewalPlusMedium(const GPSSField& field,
                                      double renewalDistance = 0.01);
    FlightState startExternal(const Point3& entry, const Vector3& direction, Random& rng) const;
    GlobalConditionalRenewalPlusSample sample(const FlightState& birth, Random& rng,
        DdaTrackingDiagnostics* diagnostics = nullptr, double maximumDistance = -1.0,
        std::uint64_t* renewalSegments = nullptr, const NumericPolicy& numeric = {}) const;
private:
    const GPSSField& field_;
    double renewalDistance_;
};

} // namespace mf

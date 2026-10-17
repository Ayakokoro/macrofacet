#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/mathutility/CompensatedSum.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {

GlobalConditionalRenewalPlusMedium::GlobalConditionalRenewalPlusMedium(
    const GPSSField& field, double renewalDistance)
    : field_(field), renewalDistance_(renewalDistance) {
    field_.validate();
    if (!(renewalDistance > 0.0) || !std::isfinite(renewalDistance))
        throw std::invalid_argument("global_conditional_renewal_plus renewal_distance must be finite and positive");
    if (!field.kernel.supportsAnalyticConditionalTransport())
        throw std::invalid_argument("global_conditional_renewal_plus requires a squared_exponential kernel");
}

FlightState GlobalConditionalRenewalPlusMedium::startExternal(const Point3& entry,
    const Vector3& direction, Random& rng) const {
    return startConditionalExterior(field_, entry, direction, rng);
}

GlobalConditionalRenewalPlusSample GlobalConditionalRenewalPlusMedium::sample(
    const FlightState& birth, Random& rng, DdaTrackingDiagnostics* diagnostics,
    double maximumDistance, std::uint64_t* renewalSegments, const NumericPolicy& numeric) const {
    if (birth.age != 0.0 || !std::isfinite(maximumDistance) ||
        !birth.birthPosition.allFinite() || !birth.birthGradient.allFinite() ||
        !(birth.birthValue >= 0.0) || !std::isfinite(birth.birthValue) ||
        !birth.direction.allFinite() || std::abs(birth.direction.squaredNorm()-1.0) > 1e-12)
        throw std::invalid_argument("invalid global_conditional_renewal_plus flight birth or distance limit");
    FlightState current = birth;
    const auto domain = field_.activeDomain.intersect({birth.birthPosition,birth.direction});
    if (!domain.hit || domain.entry > 0.0)
        throw std::invalid_argument("global_conditional_renewal_plus birth must be inside the active domain");
    const double end = maximumDistance < 0.0 ? domain.exit : std::min(maximumDistance,domain.exit);
    const ConditionalMedium medium(field_);
    CompensatedSum travelled;
    while (travelled.value() < end) {
        const double age = travelled.value();
        const double length = std::min(renewalDistance_,end-age);
        if (!(age+length > age))
            throw NumericError(NumericStatus::NeedHigherPrecision,"conditional renewal segment cannot advance");
        // Both transport modes use this same field, conditional kernel, and
        // certified majorants. Only the birth observation changes at a renewal.
        const auto flight = medium.beginFlight(current);
        if (renewalSegments) ++*renewalSegments;
        const auto hit = medium.sample(flight,rng,diagnostics,length);
        if (hit.collided)
            return {true,age+hit.age,sampleCollisionGradient(flight,hit.age,rng,numeric)};
        if (length == end-age) return {false,end,Vector3::Zero()};
        // Endpoint-positive closure, not conditioning on the full survival
        // event. Sample F and G jointly, retain the full world gradient, and
        // discard older observations. Null tracking events never renew.
        current = flight.samplePositiveEndpoint(length,rng);
        travelled.add(length);
        current.birthPosition = birth.birthPosition + travelled.value()*birth.direction;
        current.direction = birth.direction;
    }
    return {false,end,Vector3::Zero()};
}

} // namespace mf

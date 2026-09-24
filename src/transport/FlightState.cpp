#include "macrofacet/transport/FlightState.h"
#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/SmallGaussian.h"
#include <stdexcept>

namespace mf {

FlightState startExternalFlight(const Point3& x, const Vector3& w) {
    return {BirthKind::External, x, normalizedOrThrow(w), 0.0, false, Vector3::Zero()};
}

FlightState startSurfaceFlight(const Point3& x, const Vector3& g, const Vector3& w) {
    const Vector3 direction = normalizedOrThrow(w);
    if (!(g.norm() > 0.0) || !(direction.dot(g) > 0.0)) {
        throw std::invalid_argument("surface flight must depart outward from a full gradient");
    }
    return {BirthKind::Surface, x, direction, 0.0, true, g};
}

FlightState startObservedExteriorFlight(const Point3& x, double value,
                                        const Vector3& g, const Vector3& w) {
    if (!x.allFinite() || !g.allFinite() || !(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument("exterior observation requires a positive value and finite gradient");
    }
    return {BirthKind::ObservedExterior, x, normalizedOrThrow(w), 0.0, true, g, value};
}

FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w) {
    return {BirthKind::Surface, x, normalizedOrThrow(w), 0.0, false, Vector3::Zero()};
}

FlightState sampleExteriorFlight(const GPSSField& field, const Point3& x,
                                const Vector3& w, Random& rng, const NumericPolicy& policy) {
    const PointPrior prior = field.pointPrior(x);
    const double sigma = std::sqrt(prior.varianceF);
    // Stable truncated Gaussian survival inversion; F and G are independent.
    const double logTail = std::log(rng.openUniform01()) + normalLogCdf(prior.meanF / sigma);
    const double value = prior.meanF - sigma * normalQuantileFromLogCdf(logTail);
    if (!(value > 0.0) || !std::isfinite(value))
        throw NumericError(NumericStatus::NeedHigherPrecision, "positive source value lost precision");
    const Vector3 gradient = sampleGaussianPSD(Gaussian<3>{prior.meanG, prior.covarianceG}, rng, policy);
    return startObservedExteriorFlight(x, value, gradient, w);
}

void onNullCollision(FlightState& state, double delta) {
    if (!(delta >= 0.0)) throw std::invalid_argument("null-collision increment must be nonnegative");
    state.age += delta;
}

FlightState onRealSurfaceBounce(const Point3& hitPosition, const Vector3& hitGradient,
                                const Vector3& wNew) {
    return startSurfaceFlight(hitPosition, hitGradient, wNew);
}

} // namespace mf

#include "macrofacet/transport/FlightState.h"
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

FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w) {
    return {BirthKind::Surface, x, normalizedOrThrow(w), 0.0, false, Vector3::Zero()};
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


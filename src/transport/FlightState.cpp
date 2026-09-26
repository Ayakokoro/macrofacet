#include "macrofacet/transport/FlightState.h"
#include <stdexcept>

namespace mf {

FlightState startExternalFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

void onNullCollision(FlightState& state, double delta) {
    if (!(delta >= 0.0)) throw std::invalid_argument("null-collision increment must be nonnegative");
    state.age += delta;
}

} // namespace mf

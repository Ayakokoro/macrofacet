#include "macrofacet/transport/FlightState.h"
#include <stdexcept>

namespace mf {

FlightState startExternalFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

} // namespace mf

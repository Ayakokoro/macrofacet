#pragma once

#include "macrofacet/core/Types.h"

namespace mf {
struct FlightState {
    Point3 birthPosition = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    double age = 0.0;
};

FlightState startExternalFlight(const Point3& x, const Vector3& w);
FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w);
void onNullCollision(FlightState& state, double delta);

} // namespace mf

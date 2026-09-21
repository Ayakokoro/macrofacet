#pragma once

#include "macrofacet/core/Types.h"

namespace mf {

enum class BirthKind { External, Surface };
enum class ModelMode { Classic, Conditional29, Midpoint };

struct FlightState {
    BirthKind birthKind = BirthKind::External;
    Point3 birthPosition = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    double age = 0.0;
    bool hasFullGradient = false;
    Vector3 birthGradient = Vector3::Zero();
};

FlightState startExternalFlight(const Point3& x, const Vector3& w);
FlightState startSurfaceFlight(const Point3& x, const Vector3& g, const Vector3& w);
FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w);
void onNullCollision(FlightState& state, double delta);
FlightState onRealSurfaceBounce(const Point3& hitPosition, const Vector3& hitGradient,
                                const Vector3& wNew);

} // namespace mf


#pragma once

#include "macrofacet/core/Types.h"
#include "macrofacet/core/Random.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {
struct GPSSField;

enum class BirthKind { External, Surface, ObservedExterior };
enum class ExternalPolicy { OriginalMacrofacet, SampledExterior };
enum class ModelMode { Classic, Conditional29, Midpoint };

struct FlightState {
    BirthKind birthKind = BirthKind::External;
    Point3 birthPosition = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    double age = 0.0;
    bool hasFullGradient = false;
    Vector3 birthGradient = Vector3::Zero();
    double birthValue = 0.0;
};

FlightState startExternalFlight(const Point3& x, const Vector3& w);
FlightState startSurfaceFlight(const Point3& x, const Vector3& g, const Vector3& w);
FlightState startObservedExteriorFlight(const Point3& x, double value,
                                        const Vector3& g, const Vector3& w);
FlightState sampleExteriorFlight(const GPSSField& field, const Point3& x,
                                const Vector3& w, Random& rng,
                                const NumericPolicy& policy = defaultNumericPolicy());
FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w);
void onNullCollision(FlightState& state, double delta);
FlightState onRealSurfaceBounce(const Point3& hitPosition, const Vector3& hitGradient,
                                const Vector3& wNew);

} // namespace mf

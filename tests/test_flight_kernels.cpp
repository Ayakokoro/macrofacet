#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/ClassicFlightKernel.h"

void testFlightKernels(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    MaterialConfig material;
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    const FlightState state = startExternalFlight(Point3::Zero(), direction);
    ClassicFlightKernel kernel(field, material, state);
    for (double age : {0.01, 0.03, 0.08, 0.15, 0.25}) {
        const auto hazard = kernel.evaluate(age).hazard;
        context.require(hazard.value >= 0.0 && std::isfinite(hazard.value),
                        "Classic hazard is finite and nonnegative");
        const auto gradient = kernel.hitStatistics(age).collisionGradient;
        context.require(gradient.mean.allFinite() && gradient.covariance.allFinite(),
                        "Classic collision gradient is finite");
    }
}

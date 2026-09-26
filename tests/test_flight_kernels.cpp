#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

void testFlightKernels(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    MaterialConfig material;
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    const FlightState state = startExternalFlight(Point3::Zero(), direction);
    ClassicFlightKernel kernel(field, material, state);
    double previousDepth = 0.0;
    for (double age : {0.01, 0.03, 0.08, 0.15, 0.25}) {
        const auto evaluation = kernel.evaluate(age);
        context.require(evaluation.hazard.value >= 0.0, "classic flight hazard is nonnegative");
        const double depth = integrateHazard(kernel, 0.0, age).value;
        context.require(depth >= previousDepth, "classic cumulative hazard is monotone");
        previousDepth = depth;
    }
    const double end = 0.25;
    const double escape = std::exp(-integrateHazard(kernel, 0.0, end).value);
    auto density = [&](double age) {
        const double depth = age == 0.0 ? 0.0 : integrateHazard(kernel, 0.0, age).value;
        return kernel.evaluate(age).hazard.value * std::exp(-depth);
    };
    const double collisionMass = integrateFinite(density, 0.0, end).value;
    context.near(collisionMass + escape, 1.0, 3e-5,
                 "classic collision density plus escape mass normalizes");
}

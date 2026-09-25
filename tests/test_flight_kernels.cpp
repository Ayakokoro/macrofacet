#include "TestHarness.h"
#include "macrofacet/experiments/ConditionalGPReference.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

void testFlightKernels(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    const FlightState state = startSurfaceFlight(Point3::Zero(), Vector3::UnitZ(), direction);
    Conditional29FlightKernel b(field, state);
    double previousHB = 0.0;
    for (double age : {0.01, 0.03, 0.08, 0.15, 0.25}) {
        const auto be = b.evaluate(age);
        context.require(be.hazard.value >= 0.0, "flight hazard is nonnegative");
        const double hb = integrateHazard(b, 0.0, age).value;
        context.require(hb >= previousHB, "cumulative hazard is monotone");
        previousHB = hb;
        context.require(std::isfinite(*be.logExteriorScreenProbability),
                        "screen log probability remains finite");
    }
    const double end = 0.25;
    const double h = integrateHazard(b, 0.0, end).value;
    const double escape = std::exp(-h);
    auto density = [&](double t) {
        const double partial = t == 0.0 ? 0.0 : integrateHazard(b, 0.0, t).value;
        return b.evaluate(t).hazard.value * std::exp(-partial);
    };
    const double collisionMass = integrateFinite(density, 0.0, end).value;
    context.near(collisionMass + escape, 1.0, 3e-5, "flight density plus escape atom normalizes");

    const double referenceAge = 0.12;
    const auto estimate0 = estimateScreenedFirstPassageHazard(
        b.conditionedRay(), referenceAge, {}, 10000, 991);
    context.relative(estimate0.hazard, b.evaluate(referenceAge).hazard.value, 0.08,
                     "zero-checkpoint reference agrees with Eq29");
}

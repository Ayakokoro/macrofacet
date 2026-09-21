#include "TestHarness.h"
#include "macrofacet/experiments/ConditionalGPReference.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/MidpointFlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

void testFlightKernels(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    const FlightState state = startSurfaceFlight(Point3::Zero(), Vector3::UnitZ(), direction);
    Conditional29FlightKernel b(field, state);
    MidpointFlightKernel c(field, state, true);
    MidpointFlightKernel noScreen(field, state, false);
    double previousHB = 0.0;
    double previousHC = 0.0;
    for (double age : {0.01, 0.03, 0.08, 0.15, 0.25}) {
        const auto be = b.evaluate(age);
        const auto ce = c.evaluate(age);
        const auto ne = noScreen.evaluate(age);
        context.require(be.hazard.value >= 0.0 && ce.hazard.value >= 0.0,
                        "flight hazards are nonnegative");
        context.relative(ne.hazard.value, be.hazard.value, 1e-10,
                         "zero-screen midpoint exactly follows Eq29 path");
        const double hb = integrateHazard(b, 0.0, age).value;
        const double hc = integrateHazard(c, 0.0, age).value;
        context.require(hb >= previousHB && hc >= previousHC, "cumulative hazards are monotone");
        previousHB = hb;
        previousHC = hc;
        context.require(std::isfinite(*be.logExteriorScreenProbability) &&
                        std::isfinite(*ce.logExteriorScreenProbability),
                        "screen log probabilities remain finite");
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
    const auto estimate1 = estimateScreenedFirstPassageHazard(
        b.conditionedRay(), referenceAge, {0.5 * referenceAge}, 16000, 992);
    context.relative(estimate1.hazard, c.evaluate(referenceAge).hazard.value, 0.1,
                     "one-midpoint reference agrees with deterministic C");
}

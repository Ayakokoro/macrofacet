#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/FlightState.h"
#include "macrofacet/transport/MidpointFlightKernel.h"

void testSampling(TestContext& context) {
    using namespace mf;
    Random rng(1337);
    constexpr int sampleCount = 12000;
    double sampledMean = 0.0;
    const double mean = 0.2;
    const double stddev = 0.8;
    for (int i = 0; i < sampleCount; ++i) {
        const double sample = sampleNegativeFluxNormal(mean, stddev, rng);
        context.require(sample < 0.0, "negative-flux sample support");
        sampledMean += sample;
    }
    sampledMean /= sampleCount;
    // E[K under flux law] = -E[K^2 1(K<0)] / E[-K 1(K<0)].
    const double expected = -positiveRawMoment(2, -mean, stddev).value /
                            negativePartMean(mean, stddev).value;
    context.near(sampledMean, expected, 0.025, "negative-flux sample mean");

    Gaussian<3> gradient;
    gradient.mean = Vector3(0.1, -0.2, 1.0);
    gradient.covariance = (Vector3(0.3, 0.6, 0.4)).asDiagonal();
    const Vector3 w = normalizedOrThrow(Vector3(0.8, 0.1, -0.3));
    for (int i = 0; i < 1000; ++i) {
        const Vector3 g = sampleFluxWeightedGradient(gradient, w, rng);
        context.require(w.dot(g) < 0.0 && g.norm() > 0.0, "full gradient crossing support");
    }

    GPSSField field = buildDefaultField();
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    FlightState state = startSurfaceFlight(Point3::Zero(), Vector3::UnitZ(), direction);
    Conditional29FlightKernel kernel(field, state);
    for (int i = 0; i < 500; ++i) {
        const Vector3 g = sampleCollisionGradient(kernel, 0.12, rng);
        context.require(direction.dot(g) < 0.0, "conditional collision gradient crosses inward");
        const Vector3 reflected = reflectTravelDirection(direction, normalizedOrThrow(g));
        context.require(reflected.dot(g) > 0.0, "reflection departs along sampled full gradient");
    }
    MidpointFlightKernel midpoint(field, state, true);
    for (int i = 0; i < 100; ++i) {
        const Vector3 g = sampleCollisionGradient(midpoint, 0.12, rng);
        context.require(direction.dot(g) < 0.0,
                        "midpoint-screened collision gradient crosses inward");
    }

    const Point3 originalBirth = state.birthPosition;
    const Vector3 originalGradient = state.birthGradient;
    onNullCollision(state, 0.2);
    context.require((state.birthPosition - originalBirth).norm() == 0.0 &&
                    (state.birthGradient - originalGradient).norm() == 0.0 && state.age == 0.2,
                    "null collision preserves statistical birth state");
    const FlightState reset = onRealSurfaceBounce(Point3(1.0, 0.0, 0.0), Vector3::UnitX(),
                                                  Vector3::UnitX());
    context.require(reset.age == 0.0 && reset.hasFullGradient &&
                    (reset.birthGradient - Vector3::UnitX()).norm() == 0.0,
                    "real bounce resets age and stores full gradient");
}

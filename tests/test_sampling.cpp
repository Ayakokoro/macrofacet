#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include "macrofacet/transport/FlightState.h"
#include "macrofacet/transport/NarrowBandMedium.h"

namespace {

class UnitDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 1.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override { return {1.0, 1.0, true}; }
};

class CountedDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 1.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override {
        if (forbidBounds) throw std::runtime_error("unexpected density bounds query during flight");
        ++boundsCalls;
        return {1.0, 1.0, true};
    }
    mutable int boundsCalls = 0;
    bool forbidBounds = false;
};

void testPrecomputedDdaBounds(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    auto density = std::make_shared<CountedDensity>();
    auto grid = std::make_shared<DensityMajorantGrid>(*density, field.activeDomain, 4);
    context.require(density->boundsCalls == 4 * 4 * 4,
                    "DDA density bounds are certified during grid construction");
    density->forbidBounds = true;
    MaterialConfig material;
    NarrowBandMedium medium(field, material, density, true, grid);
    Random rng(419);
    for (int i = 0; i < 8; ++i) {
        const FlightState state = startExternalFlight(Point3(0.0, 0.0, 0.2),
                                                      -Vector3::UnitZ());
        const auto flight = medium.beginFlight(ModelMode::Classic, state,
                                               ExternalPolicy::OriginalMacrofacet,
                                               defaultNumericPolicy());
        (void)medium.sample(*flight, rng, defaultNumericPolicy(), nullptr, 8);
    }
    context.require(density->boundsCalls == 4 * 4 * 4,
                    "DDA flights reuse precomputed density bounds");

    auto fallbackDensity = std::make_shared<CountedDensity>();
    NarrowBandMedium fallback(field, material, fallbackDensity, true);
    context.require(fallbackDensity->boundsCalls == 1,
                    "fallback density bound is certified during medium construction");
    fallbackDensity->forbidBounds = true;
    for (int i = 0; i < 8; ++i) {
        const FlightState state = startExternalFlight(Point3(0.0, 0.0, 0.2),
                                                      -Vector3::UnitZ());
        const auto flight = fallback.beginFlight(ModelMode::Classic, state,
                                                 ExternalPolicy::OriginalMacrofacet,
                                                 defaultNumericPolicy());
        (void)fallback.sample(*flight, rng, defaultNumericPolicy(), nullptr, 8);
    }
    context.require(fallbackDensity->boundsCalls == 1,
                    "fallback flights reuse the cached density bound");
}

void testDdaBoundaryExit(TestContext& context) {
    using namespace mf;
    const Bounds3 domain{
        Point3(-0.052075867396115624, -0.052069087367772424, -0.007454654396949074),
        Point3(0.05219135943238215, 0.05219813946072535, 0.08912072782944641)};
    UnitDensity density;
    DensityMajorantGrid grid(density, domain, 64);
    for (int i = 0; i < 24; ++i) {
        const Point3 origin(-0.0283247 + 0.0015 * i,
                            0.0399771 - 0.002 * i, domain.maximum.z());
        const Vector3 direction = normalizedOrThrow(
            Vector3(-0.581425 - 0.003 * i, 0.780764 - 0.002 * i, -0.228807));
        const Ray ray{origin, direction};
        const DomainInterval interval = domain.intersect(ray);
        context.require(interval.hit && interval.exit > interval.entry,
                        "oblique DDA ray crosses the domain");
        const auto segments = grid.segments(ray, interval.entry, interval.exit);
        context.require(!segments.empty() && segments.front().begin == interval.entry &&
                        segments.back().end == interval.exit,
                        "DDA covers the ray exactly through the outer domain face");
    }
}

} // namespace

void testSampling(TestContext& context) {
    using namespace mf;
    testDdaBoundaryExit(context);
    testPrecomputedDdaBounds(context);
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

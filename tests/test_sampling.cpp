#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
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

class CountedAlpha final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 0.5; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override {
        if (forbidBounds) throw std::runtime_error("unexpected alpha bounds query during flight");
        ++boundsCalls;
        return {0.5, 0.5, true};
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
    auto alpha = std::make_shared<CountedAlpha>();
    material.alphaField = alpha;
    NarrowBandMedium medium(field, material, density, grid);
    context.require(alpha->boundsCalls == 1,
                    "local area bound is baked once when the medium is prepared");
    alpha->forbidBounds = true;
    Random rng(419);
    for (int i = 0; i < 8; ++i) {
        const FlightState state = startExternalFlight(Point3(0.0, 0.0, 0.2),
                                                      -Vector3::UnitZ());
        const auto flight = medium.beginFlight(state);
        (void)medium.sample(flight, rng, defaultNumericPolicy());
    }
    context.require(density->boundsCalls == 4 * 4 * 4,
                    "DDA flights reuse precomputed density bounds");
    context.require(alpha->boundsCalls == 1,
                    "DDA flights reuse the precomputed local area bound");


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
        auto cursor = grid.sampleRay(ray, interval.entry, interval.exit);
        for (const auto& expected : segments) {
            const auto actual=cursor.next();
            context.require(actual && actual->begin==expected.begin &&
                actual->end==expected.end &&
                actual->densityMaximum==expected.densityMaximum,
                "lazy density DDA reproduces the compatibility segment sequence");
        }
        context.require(!cursor.next(),"lazy density DDA ends after its final cell");
        context.require(!segments.empty() && segments.front().begin == interval.entry &&
                        segments.back().end == interval.exit,
                        "DDA covers the ray exactly through the outer domain face");
    }
}

void testDdaFlightMass(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    MaterialConfig material;
    auto density = std::make_shared<UnitDensity>();
    auto grid = std::make_shared<DensityMajorantGrid>(*density, field.activeDomain, 8);
    NarrowBandMedium medium(field, material, density, grid);
    const FlightState state = startExternalFlight(Point3::Zero(), Vector3::UnitX());
    const ClassicFlightKernel flight = medium.beginFlight(state);
    const double hazard = classicProjectedArea(field, material, Point3::Zero(),
                                               Vector3::UnitX()).value;
    const double expectedEscape = std::exp(-hazard);
    Random rng(881);
    DdaTrackingDiagnostics diagnostics;
    int escapes = 0;
    constexpr int samples = 5000;
    for (int i = 0; i < samples; ++i) {
        const FlightSample result = medium.sample(flight, rng, defaultNumericPolicy(),
                                                  &diagnostics, 1.0);
        if (!result.collided) ++escapes;
        context.require(result.age >= 0.0 && result.age <= 1.0,
                        "DDA flight remains in its interval");
    }
    context.near(double(escapes) / samples, expectedEscape, 0.025,
                 "DDA collision and escape probabilities normalize");
    context.require(diagnostics.candidates >= static_cast<std::uint64_t>(samples - escapes),
                    "DDA diagnostics count collision candidates");
}

} // namespace

void testSampling(TestContext& context) {
    using namespace mf;
    testDdaBoundaryExit(context);
    testPrecomputedDdaBounds(context);
    testDdaFlightMass(context);
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

    for (double sigma : {1e-12,1e-20}) {
        Random concentratedRng(773);
        double gammaMean=0.0, centeredMean=0.0;
        for (int i=0; i<2000; ++i) {
            const double negative=sampleNegativeFluxNormal(-1.0,sigma,concentratedRng);
            context.require(negative<0.0 && std::isfinite(negative),
                            "concentrated negative flux does not lose its standardized CDF");
            centeredMean+=(negative+1.0)/sigma;
            const double positive=sampleNegativeFluxNormal(1.0,sigma,concentratedRng);
            context.require(positive<0.0 && std::isfinite(positive),
                            "far positive-tail flux sampler stays on negative support");
            gammaMean+=-positive/(sigma*sigma);
        }
        context.near(centeredMean/2000.0,0.0,0.12,"narrow negative flux retains its Gaussian limit");
        context.near(gammaMean/2000.0,2.0,0.15,"positive-tail flux retains its Gamma(2) scaling limit");
    }

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
    FlightState state = startExternalFlight(Point3::Zero(), direction);
    MaterialConfig material;
    ClassicFlightKernel kernel(field, material, state);
    for (int i = 0; i < 500; ++i) {
        const Vector3 g = sampleCollisionGradient(kernel, 0.12, rng);
        context.require(direction.dot(g) < 0.0, "classic collision gradient crosses inward");
        const Vector3 reflected = reflectTravelDirection(direction, normalizedOrThrow(g));
        context.require(reflected.dot(g) > 0.0, "reflection departs along sampled full gradient");
    }

    const FlightState reset = startClassicCollisionFlight(Point3(1.0, 0.0, 0.0),
                                                            Vector3::UnitX());
    context.require(reset.age == 0.0 &&
                    (reset.birthPosition - Vector3::UnitX()).norm() == 0.0,
                    "classic bounce resets flight age and origin");

}

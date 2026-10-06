#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <memory>
#include <stdexcept>

namespace {

bool sameStatistics(const mf::RenderStatistics& a, const mf::RenderStatistics& b) {
    return a.paths == b.paths && a.realCollisions == b.realCollisions &&
           a.escapedPaths == b.escapedPaths &&
           a.rouletteTerminations == b.rouletteTerminations &&
           a.safetyCapTerminations == b.safetyCapTerminations &&
           a.numericalFailures == b.numericalFailures &&
           a.accumulatedPathDepth == b.accumulatedPathDepth &&
           a.tracking.candidates == b.tracking.candidates &&
           a.tracking.nullCollisions == b.tracking.nullCollisions &&
           a.tracking.boundIntervals == b.tracking.boundIntervals &&
           a.tracking.nearCandidates == b.tracking.nearCandidates &&
           a.tracking.farCandidates == b.tracking.farCandidates &&
           a.tracking.roundedCandidateSteps == b.tracking.roundedCandidateSteps &&
           a.tracking.adaptiveMajorantFlights == b.tracking.adaptiveMajorantFlights &&
           a.tracking.trackingSegments == b.tracking.trackingSegments &&
           a.tracking.maximumSegmentNullCollisions == b.tracking.maximumSegmentNullCollisions &&
           a.tracking.maximumMajorant == b.tracking.maximumMajorant &&
           a.tracking.maximumCandidateMajorant == b.tracking.maximumCandidateMajorant &&
           a.tracking.maximumEvaluatedHazard == b.tracking.maximumEvaluatedHazard &&
           a.tracking.maximumSegmentOpticalDepth == b.tracking.maximumSegmentOpticalDepth &&
           a.tracking.maximumConditionalIntervalMajorant ==
               b.tracking.maximumConditionalIntervalMajorant;
}

class ThrowingMean final : public mf::MeanField {
public:
    const char* typeName() const override { return "throwing_test_stub"; }
    mf::MeanJet evaluate(const mf::Point3&) const override {
        throw std::runtime_error("intentional worker failure");
    }
    mf::BoundsSummary bounds(const mf::Bounds3&) const override {
        return {-1.0, 1.0, true};
    }
};

class UnitDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 1.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override { return {1.0, 1.0, true}; }
};

class CountedAlpha final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 0.5; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override {
        ++boundsCalls;
        return {0.5, 0.5, true};
    }
    mutable int boundsCalls = 0;
};

} // namespace

void testRenderingThreads(TestContext& context) {
    using namespace mf;
    ExperimentConfig config;
    config.field = buildDefaultField();
    config.render.width = 3;
    config.render.height = 3;
    config.render.samplesPerPixel = 1;
    config.mediumDensity = std::make_shared<UnitDensity>();
    config.densityMajorantGrid = std::make_shared<DensityMajorantGrid>(
        *config.mediumDensity, config.field.activeDomain, 4);
    auto alpha = std::make_shared<CountedAlpha>();
    config.material.alphaField = alpha;
    config.render.environment = "directional_gradient";
    {
        config.render.threadCount = 1;
        std::atomic<std::uint64_t> completedRays{0};
        const RenderedImage serial = renderAnalyticScene(config, &completedRays);
        context.require(completedRays.load()==static_cast<std::uint64_t>(
            config.render.width*config.render.height*config.render.samplesPerPixel),
            "single-thread render reports each completed camera ray");
        context.require(alpha->boundsCalls == 1,
                        "serial render bakes the alpha area bound once for all paths");
        config.render.threadCount = 3;
        completedRays.store(0);
        const RenderedImage parallel = renderAnalyticScene(config, &completedRays);
        context.require(completedRays.load()==static_cast<std::uint64_t>(
            config.render.width*config.render.height*config.render.samplesPerPixel),
            "parallel render reports each completed camera ray");
        context.require(alpha->boundsCalls == 2,
                        "parallel render shares one baked area bound across workers");
        context.require(serial.pixels.size() == parallel.pixels.size(),
                        "threaded render pixel count");
        bool pixelsEqual = serial.pixels.size() == parallel.pixels.size();
        for (std::size_t i = 0; pixelsEqual && i < serial.pixels.size(); ++i) {
            pixelsEqual = (serial.pixels[i].array() == parallel.pixels[i].array()).all();
        }
        context.require(pixelsEqual, "render pixels are independent of worker count");
        context.require(sameStatistics(serial.statistics, parallel.statistics),
                        "render statistics are independent of worker count");
        config.preparedAreaMajorant = 2.0;
        (void)renderAnalyticScene(config);
        context.require(alpha->boundsCalls == 2,
                        "a prepared render reuses the baked alpha bound without scanning the field");
        config.preparedAreaMajorant.reset();
    }

    config.material.alphaField.reset();
    config.transportMode = "global_conditional";
    config.render.threadCount = 1;
    const RenderedImage conditionalSerial = renderAnalyticScene(config);
    config.render.threadCount = 3;
    const RenderedImage conditionalParallel = renderAnalyticScene(config);
    bool conditionalEqual = conditionalSerial.pixels.size() == conditionalParallel.pixels.size();
    for (std::size_t i = 0; conditionalEqual && i < conditionalSerial.pixels.size(); ++i)
        conditionalEqual = (conditionalSerial.pixels[i].array() ==
                            conditionalParallel.pixels[i].array()).all();
    context.require(conditionalEqual,
                    "conditional render pixels are independent of worker count");
    context.require(sameStatistics(conditionalSerial.statistics,
                                   conditionalParallel.statistics),
                    "conditional render statistics are independent of worker count");
    config.transportMode = "classic";

    config.field.mean = std::make_shared<ThrowingMean>();
    config.render.threadCount = 3;
    bool propagated = false;
    try {
        (void)renderAnalyticScene(config);
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    context.require(propagated, "worker exception is joined and propagated");
}

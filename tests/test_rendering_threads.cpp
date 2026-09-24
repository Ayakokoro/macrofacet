#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <array>
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
           a.tracking.hazardEvaluations == b.tracking.hazardEvaluations &&
           a.tracking.newtonIterations == b.tracking.newtonIterations &&
           a.tracking.bisectionSteps == b.tracking.bisectionSteps &&
           a.tracking.maximumResidual == b.tracking.maximumResidual;
}

class ThrowingMean final : public mf::MeanField {
public:
    mf::MeanJet evaluate(const mf::Point3&) const override {
        throw std::runtime_error("intentional worker failure");
    }
    mf::BoundsSummary bounds(const mf::Bounds3&) const override {
        return {-1.0, 1.0, true};
    }
};

} // namespace

void testRenderingThreads(TestContext& context) {
    using namespace mf;
    ExperimentConfig config;
    config.field = buildDefaultField();
    config.render.width = 3;
    config.render.height = 3;
    config.render.samplesPerPixel = 1;
    config.render.flightTableCells = 4;
    config.render.environment = "directional_gradient";
    config.conditional29.externalPolicy = ExternalPolicy::SampledExterior;

    for (ModelMode mode : std::array<ModelMode, 3>{ModelMode::Classic,
                                                    ModelMode::Conditional29,
                                                    ModelMode::Midpoint}) {
        config.render.threadCount = 1;
        const RenderedImage serial = renderAnalyticScene(mode, config);
        config.render.threadCount = 3;
        const RenderedImage parallel = renderAnalyticScene(mode, config);
        context.require(serial.pixels.size() == parallel.pixels.size(),
                        "threaded render pixel count");
        bool pixelsEqual = serial.pixels.size() == parallel.pixels.size();
        for (std::size_t i = 0; pixelsEqual && i < serial.pixels.size(); ++i) {
            pixelsEqual = (serial.pixels[i].array() == parallel.pixels[i].array()).all();
        }
        context.require(pixelsEqual, "render pixels are independent of worker count");
        context.require(sameStatistics(serial.statistics, parallel.statistics),
                        "render statistics are independent of worker count");
    }

    config.field.mean = std::make_shared<ThrowingMean>();
    config.render.threadCount = 3;
    bool propagated = false;
    try {
        (void)renderAnalyticScene(ModelMode::Classic, config);
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    context.require(propagated, "worker exception is joined and propagated");
}

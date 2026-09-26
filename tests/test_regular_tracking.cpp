#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/RootFinding.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include <functional>

namespace {
class AnalyticKernel final : public mf::FlightKernel {
public:
    AnalyticKernel(const mf::GPSSField& field, mf::FlightState state,
                   std::function<double(double)> rate)
        : FlightKernel(field, state), rate_(std::move(rate)) {}
    mf::HazardEvaluation evaluate(double t) const override {
        const double rate = rate_(t);
        return {rate > 0 ? mf::positiveFromLog(std::log(rate)) : mf::exactZero()};
    }
    mf::HitStatistics hitStatistics(double) const override { return {}; }
private:
    std::function<double(double)> rate_;
};


}

void testRegularTracking(TestContext& context) {
    using namespace mf;
    NumericPolicy policy;
    policy.relativeTolerance = 1e-8;
    policy.absoluteTolerance = 1e-10;
    policy.distanceAbsoluteTolerance = 1e-12;
    policy.distanceRelativeTolerance = 1e-11;
    GPSSField field = buildDefaultField();
    const FlightState source = startExternalFlight(Point3::Zero(), Vector3::UnitX());
    AnalyticKernel constant(field, source, [](double) { return 2.0; });
    TrackingDiagnostics diagnostics;
    OpticalDepthSampler constantSampler(constant, policy, 1.0, 4, &diagnostics);
    const auto hit = constantSampler.invert(0.7);
    context.near(hit.age, 0.35, 1e-10, "regular tracking inverts constant extinction");
    context.near(*hit.logDistancePdf, std::log(2.0) - 0.7, 1e-10, "hit PDF uses continuous hazard");
    const auto escape = constantSampler.invert(3.0);
    context.require(!escape.collided && escape.age == 1.0, "escape is the domain boundary event");
    context.near(*escape.escapeMass, std::exp(-2.0), 1e-12, "escape mass normalizes flight law");
    (void)constantSampler.invert(0.73123);
    context.require(diagnostics.newtonIterations > 0, "actual sampler invokes Newton");

    AnalyticKernel linear(field, source, [](double t) { return 2.0 * t; });
    OpticalDepthSampler linearSampler(linear, policy, 1.0, 4);
    for (double e : {1e-9, 0.017, 0.36, 0.999999}) {
        const auto sample = linearSampler.invert(e);
        context.near(sample.age, std::sqrt(e), 2e-9, "nonconstant hazard is not linearly interpolated");
        context.near(-*sample.logSurvival, sample.age * sample.age, 1e-10, "reported survival uses actual integral");
    }
    FlightState advanced = source;
    advanced.age = 0.2;
    AnalyticKernel remaining(field, advanced, [](double t) { return 2.0 * t; });
    OpticalDepthSampler rest(remaining, policy, 1.0);
    context.near(rest.invert(0.32).age, 0.6, 1e-9, "resumed flight keeps absolute birth age");
    AnalyticKernel zero(field, source, [](double) { return 0.0; });
    context.near(*OpticalDepthSampler(zero, policy, 1.0).invert(0.2).escapeMass, 1.0, 0.0,
                 "zero hazard has unit escape mass");

    int fallbacks = 0;
    const auto root = solveMonotoneSafeguardedNewton([](double x) {
        const double y = std::max(0.0, x - 0.4);
        return RootEvaluation{y * y, 2.0 * y};
    }, 0.25, 0.0, 1.0, 0.1, policy, &fallbacks);
    context.require(root.status == NumericStatus::Ok && fallbacks > 0, "Newton safely leaves a zero-derivative plateau");
    context.near(root.value, 0.9, 1e-9, "safeguarded Newton plateau root");
    NumericPolicy exhausted = policy;
    exhausted.maxRootIterations = 0;
    bool failed = false;
    try { (void)OpticalDepthSampler(linear, exhausted, 1.0).invert(0.123); }
    catch (const NumericError&) { failed = true; }
    context.require(failed, "root budget exhaustion is not reported as escape");

    Random rng(7123);
    int belowHalf = 0, escapes = 0;
    constexpr int count = 6000;
    for (int i = 0; i < count; ++i) {
        const auto sample = linearSampler.sample(rng);
        if (!sample.collided) ++escapes;
        else if (sample.age <= 0.5) ++belowHalf;
    }
    context.near(double(belowHalf) / count, -std::expm1(-0.25), 0.02, "sampled distance CDF");
    context.near(double(escapes) / count, std::exp(-1.0), 0.02, "sampled escape frequency");
    {
        Random narrow(917), ordinary(917);
        for (int i = 0; i < 64; ++i) {
            const double tiny = sampleNegativeFluxNormal(0.2e-12, 0.8e-12, narrow, policy);
            const double full = sampleNegativeFluxNormal(0.2, 0.8, ordinary, policy);
            context.near(tiny / 1e-12, full, 1e-8, "flux sampling is invariant to tiny gradient scales");
        }
        const double tail = sampleNegativeFluxNormal(40.0, 1.0, rng, policy);
        context.require(tail < 0.0 && std::isfinite(tail), "flux sampling survives an underflowed normalizer");
        const double tailCdf = negativeFluxNormalCdf(tail, 40.0, 1.0, policy).value;
        context.require(tailCdf > 0.0 && tailCdf < 1.0, "rare flux tail retains a nontrivial CDF");
    }
}

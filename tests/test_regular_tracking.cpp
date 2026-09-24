#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/gpss/ConditionedRay.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/mathutility/RootFinding.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/ConditionalConductorPhase.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include <functional>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
class AnalyticKernel final : public mf::FlightKernel {
public:
    AnalyticKernel(const mf::GPSSField& field, mf::FlightState state,
                   std::function<double(double)> rate)
        : FlightKernel(field, state), rate_(std::move(rate)) {}
    mf::HazardEvaluation evaluate(double t) const override {
        const double rate = rate_(t);
        return {rate > 0 ? mf::positiveFromLog(std::log(rate)) : mf::exactZero(), {}, {}};
    }
    mf::HitStatistics hitStatistics(double) const override { return {}; }
    mf::ModelMode mode() const override { return mf::ModelMode::Classic; }
private:
    std::function<double(double)> rate_;
};

mf::Gaussian<4> matrixReference(const mf::GPSSField& field, const mf::Point3& x,
                                double f, const mf::Vector3& g, const mf::Vector3& d, double t) {
    using namespace mf;
    const auto p0 = field.pointPrior(x), pt = field.pointPrior(x + t * d);
    Gaussian<4> observation, target;
    observation.mean << p0.meanF, p0.meanG;
    target.mean << pt.meanF, pt.meanG;
    observation.covariance(0, 0) = p0.varianceF;
    observation.covariance.bottomRightCorner<3, 3>() = p0.covarianceG;
    target.covariance(0, 0) = pt.varianceF;
    target.covariance.bottomRightCorner<3, 3>() = pt.covarianceG;
    const auto jet = field.kernel.evaluate(x + t * d, x);
    Eigen::Matrix4d cross;
    cross(0, 0) = jet.valueValue;
    cross.topRightCorner<1, 3>() = jet.valueXGradientY.transpose();
    cross.bottomLeftCorner<3, 1>() = jet.gradientXValueY;
    cross.bottomRightCorner<3, 3>() = jet.gradientXGradientY;
    Eigen::Vector4d observed;
    observed << f, g;
    return conditionGaussian(target, observation, cross, observed);
}
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

    const Vector3 d = normalizedOrThrow(Vector3(0.9, 0.1, 0.3));
    const Point3 x(0.0, 0.0, 0.02);
    const Vector3 g(0.1, -0.1, 1.0);
    for (bool sphere : {false, true}) {
        if (sphere) field.mean = std::make_shared<SphereMean>(Point3(0.0, 0.0, -0.7), 0.7);
        const Matrix3 rotation = Eigen::AngleAxisd(0.5, Vector3::UnitY()).toRotationMatrix();
        field.kernel = SquaredExponentialKernel::fromCorrelationLengths(0.1, Vector3(0.2, 0.3, 0.4), rotation);
        for (double f0 : {0.0, 0.12}) {
            ConditionedRay ray(field, x, f0, g, d, policy);
            for (double t : {0.04, 0.2, 0.8}) {
                const auto reference = matrixReference(field, x, f0, g, d, t);
                const auto analytic = ray.endpointValueGradient(t);
                context.near((reference.mean - analytic.mean).norm(), 0.0, 1e-11, "SE means match joint Gaussian conditioning");
                context.near((reference.covariance - analytic.covariance).norm(), 0.0, 1e-11, "SE covariance matches joint Gaussian conditioning");
                Gaussian<3> target{reference.mean.tail<3>(), reference.covariance.bottomRightCorner<3, 3>()};
                Gaussian<1> observation;
                observation.mean[0] = reference.mean[0];
                observation.covariance(0, 0) = reference.covariance(0, 0);
                const Eigen::Vector3d cross = reference.covariance.bottomLeftCorner<3, 1>();
                const auto conditioned = conditionGaussian(target, observation, cross, Eigen::Matrix<double, 1, 1>::Zero());
                const auto stable = ray.gradientGivenEndpointZero(t);
                context.near((conditioned.mean - stable.mean).norm(), 0.0, 2e-9, "stable zero-value gradient mean");
                context.near((conditioned.covariance - stable.covariance).norm(), 0.0, 2e-11, "stable Schur complement covariance");
                const auto k = ray.slopeGivenEndpointZero(t);
                context.near(d.dot(stable.mean), k.mean[0], 2e-11, "scalar and full gradient means agree");
                context.near(d.dot(stable.covariance * d), k.covariance(0, 0), 2e-12, "scalar and full gradient variances agree");
            }
        }
    }

    field = buildDefaultField();
    ConditionedRay near(field, Point3::Zero(), Vector3::UnitZ(), Vector3::UnitZ());
    for (double t : {1e-8, 1e-6, 1e-4}) {
        const double z = 25.0 * t * t;
        const double expected = 0.01 * 25.0 * z * z / 6.0;
        context.near(near.slopeGivenEndpointZero(t).covariance(0, 0) / expected, 1.0, 1e-6,
                     "small positive Schur variance survives cancellation");
    }
    Conditional29FlightKernel grazing(field, startSurfaceFlight(Point3::Zero(), Vector3(1e-6, 0.0, 1.0), Vector3::UnitX()), policy);
    context.require(grazing.evaluate(4e-7).hazard.value > 0.0, "small positive field variance does not erase grazing collisions");
    {
        OpticalDepthSampler a(grazing, policy, 0.4, 4), b(grazing, policy, 0.4, 64);
        const auto first = a.invert(0.2), second = b.invert(0.2);
        context.require(first.collided && second.collided, "grazing collision survives initial panel changes");
        context.near(first.age, second.age, 1e-11, "initial panel count does not define the collision model");
    }
    {
        SphereMean sphere(Point3::Zero(), 1.0);
        context.near(sphere.valueDifference(Vector3::UnitZ(), Vector3(1e-9, 0, 0)) / 5e-19,
                     1.0, 1e-12, "sphere mean retains tiny tangential height changes");
    }

    const auto incoming = startObservedExteriorFlight(Point3(0, 0, 0.1), 0.08, Vector3::UnitZ(), -Vector3::UnitZ());
    Conditional29FlightKernel exterior(field, incoming, policy);
    const auto e = OpticalDepthSampler(exterior, policy).invert(0.5);
    context.require(e.collided && e.age > 0.0, "positive source permits an inward initial slope");
    const double actual = integrateHazard(exterior, 0.0, e.age, policy).value;
    context.near(actual, 0.5, 2e-8, "conditional distance independently reintegrates to its target");
    {
        const auto gradient = exterior.hitStatistics(e.age + 0.15).gradientGivenEndpointZero;
        ConductorParameters unit;
        unit.forceUnitFresnel = true;
        const ConditionalConductorPhase phase(gradient, unit, policy);
        const GaussianNdf ndf(gradient.mean, gradient.covariance, policy);
        double directionIntegral = 0.0, projectedIntegral = 0.0;
        constexpr int sphereSamples = 20000;
        for (int i = 0; i < sphereSamples; ++i) {
            const double z = 1.0 - 2.0 * (i + 0.5) / sphereSamples;
            const double phi = i * 2.39996322972865332;
            const double radius = std::sqrt(1.0 - z * z);
            const Vector3 v(radius * std::cos(phi), radius * std::sin(phi), z);
            const double pdf = phase.evaluateSamplingPdf(incoming.direction, v);
            directionIntegral += pdf;
            projectedIntegral += std::max(0.0, -incoming.direction.dot(v)) * ndf.evaluateD(v).value;
            if (i == 123)
                context.near(phase.evaluateEnergy(incoming.direction, v).x(), pdf, 1e-12,
                             "unit Fresnel conditional scattering kernel equals geometric PDF");
        }
        context.near(directionIntegral * 4.0 * kPi / sphereSamples, 1.0, 0.002,
                     "conditional reflected-direction PDF normalizes");
        context.near(projectedIntegral * 4.0 * kPi / sphereSamples,
                     ndf.projectedArea(incoming.direction).value, 0.001,
                     "conditional NDF projected area matches hazard flux");
    }
    double meanValue = 0.0;
    for (int i = 0; i < 2000; ++i) {
        const auto s = sampleExteriorFlight(field, Point3::Zero(), -Vector3::UnitZ(), rng, policy);
        context.require(s.birthValue > 0.0 && s.hasFullGradient, "exterior source has complete positive observations");
        meanValue += s.birthValue;
    }
    context.near(meanValue / 2000, 0.1 * std::sqrt(2.0 / kPi), 0.004, "positive source has truncated Gaussian mean");

    // The renderer must propagate numeric failure, not silently return black.
    ExperimentConfig config;
    config.field = field;
    config.numeric.maxRootIterations = 0;
    config.conditional29.externalPolicy = ExternalPolicy::SampledExterior;
    RenderStatistics statistics;
    failed = false;
    try { (void)traceCameraPath({Point3(0, 0, 1), -Vector3::UnitZ()}, ModelMode::Conditional29, config, rng, statistics); }
    catch (const NumericError&) { failed = true; }
    context.require(failed, "renderer propagates failed conditional inversion");
    config.numeric = policy;
    config.conditional29.externalPolicy = ExternalPolicy::OriginalMacrofacet;
    statistics = {};
    (void)traceCameraPath({Point3(0, 0, 1), -Vector3::UnitZ()}, ModelMode::Conditional29, config, rng, statistics);
    context.require(statistics.tracking.newtonIterations > 0 && statistics.numericalFailures == 0,
                    "legacy external model also uses regular tracking for conditional29");

    const auto project = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto ci = loadExperimentConfig(project / "configs/macrofacet_ci.json");
    context.require(ci.conditional29.externalPolicy == ExternalPolicy::SampledExterior &&
                    !ci.conditional29.hardDepthCap, "CI selects positive exterior and unlimited conditional bounces");
    const auto temporary = project / "build/regular_tracking_config_test.json";
    writeResolvedConfig(ci, temporary);
    nlohmann::json resolved;
    { std::ifstream stream(temporary); stream >> resolved; }
    context.require(resolved["transport"]["conditional29"]["sampler"] == "regular_tracking" &&
                    resolved["transport"]["conditional29"]["optical_depth_solver"] == "safeguarded_newton",
                    "resolved config reports actual rendering algorithm");
    nlohmann::json input;
    { std::ifstream stream(project / "configs/macrofacet_ci.json"); stream >> input; }
    input["transport"]["conditional29"]["sampler"] = "delta_tracking";
    { std::ofstream stream(temporary); stream << input; }
    failed = false;
    try { (void)loadExperimentConfig(temporary); }
    catch (const std::invalid_argument&) { failed = true; }
    context.require(failed, "conditional29 refuses an unsupported tracking algorithm");
    std::filesystem::remove(temporary);
}

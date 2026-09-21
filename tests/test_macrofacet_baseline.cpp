#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

void testMacrofacetBaseline(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    const PointPrior prior = field.pointPrior(Point3::Zero());
    GaussianNdf ndf(prior.meanG, prior.covarianceG);
    const Vector3 w = normalizedOrThrow(Vector3(0.7, 0.1, -0.5));
    constexpr int sampleCount = 40000;
    Vector3 integral = Vector3::Zero();
    double projected = 0.0;
    double visibleMass = 0.0;
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < sampleCount; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / sampleCount;
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = golden * i;
        const Vector3 normal(radius * std::cos(phi), radius * std::sin(phi), z);
        const double d = ndf.evaluateD(normal).value;
        integral += normal * d;
        projected += std::max(-w.dot(normal), 0.0) * d;
        visibleMass += ndf.visibleNormalPdf(normal, w);
    }
    integral *= 4.0 * kPi / sampleCount;
    projected *= 4.0 * kPi / sampleCount;
    visibleMass *= 4.0 * kPi / sampleCount;
    context.require((integral - prior.meanG).norm() < 0.015, "oriented-area NDF identity");
    context.relative(projected, ndf.projectedArea(w).value, 0.015,
                     "Gaussian projected-area identity");
    context.near(visibleMass, 1.0, 0.015, "VNDF normalization");

    GPSSField unitField = field;
    unitField.conductor.forceUnitFresnel = true;
    ConductorPhase uniformPhase(unitField, Point3::Zero(), 0.0);
    ConductorPhase mixedPhase(unitField, Point3::Zero(), 0.5);
    ConductorPhase targetPhase(unitField, Point3::Zero(), 0.0, true);
    double energyMass = 0.0;
    double uniformProposalMass = 0.0;
    double mixedProposalMass = 0.0;
    double targetProposalMass = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / sampleCount;
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = golden * i;
        const Vector3 outgoing(radius * std::cos(phi), radius * std::sin(phi), z);
        energyMass += uniformPhase.evaluateEnergy(w, outgoing).x();
        uniformProposalMass += uniformPhase.evaluateSamplingPdf(w, outgoing);
        mixedProposalMass += mixedPhase.evaluateSamplingPdf(w, outgoing);
        targetProposalMass += targetPhase.evaluateSamplingPdf(w, outgoing);
    }
    energyMass *= 4.0 * kPi / sampleCount;
    uniformProposalMass *= 4.0 * kPi / sampleCount;
    mixedProposalMass *= 4.0 * kPi / sampleCount;
    targetProposalMass *= 4.0 * kPi / sampleCount;
    context.near(energyMass, 1.0, 0.02, "unit-Fresnel phase energy normalization");
    context.near(uniformProposalMass, 1.0, 0.02, "uniform phase proposal normalization");
    context.near(mixedProposalMass, 1.0, 0.025, "mixed phase proposal normalization");
    context.near(targetProposalMass, 1.0, 0.02, "target VNDF proposal normalization");
    Random targetRng(8801);
    for (int i = 0; i < 64; ++i) {
        const PhaseSample sample = targetPhase.samplePhase(w, targetRng);
        context.require((sample.throughputWeight - Spectrum::Ones()).norm() < 1e-10,
                        "unit-Fresnel target VNDF has unit sample weight");
    }

    GgxHeightfield ggx(0.5, 0.8);
    double ggxProjected = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / sampleCount;
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = golden * i;
        const Vector3 normal(radius * std::cos(phi), radius * std::sin(phi), z);
        ggxProjected += std::max(-w.dot(normal), 0.0) * ggx.evaluateD(normal).value;
    }
    ggxProjected *= 4.0 * kPi / sampleCount;
    context.relative(ggxProjected, ggx.projectedArea(w).value, 0.02, "GGX projected area");

    ConductorParameters unit;
    unit.forceUnitFresnel = true;
    context.require((conductorFresnel(0.3, unit) - Spectrum::Ones()).norm() == 0.0,
                    "unit Fresnel test branch");
    ConductorParameters material;
    const Spectrum normalFresnel = conductorFresnel(1.0, material);
    for (int i = 0; i < 3; ++i) {
        const double expected = (std::pow(material.eta[i] - 1.0, 2) + material.k[i] * material.k[i]) /
                                (std::pow(material.eta[i] + 1.0, 2) + material.k[i] * material.k[i]);
        context.near(normalFresnel[i], expected, 2e-14, "normal-incidence conductor Fresnel");
    }

    const FlightState external = startExternalFlight(Point3(0.0, 0.0, 0.2),
                                                     normalizedOrThrow(Vector3(0.6, 0.0, -0.8)));
    ClassicFlightKernel kernel(field, external);
    const double end = std::min(0.3, kernel.maximumAgeInDomain());
    const double numerical = integrateHazard(kernel, 0.0, end).value;
    const double area = ndf.projectedArea(external.direction).value;
    const double d0 = 0.2;
    const double wz = external.direction.z();
    const double analytic = area / wz *
        (normalLogCdf((d0 + wz * end) / field.kernel.sigma()) -
         normalLogCdf(d0 / field.kernel.sigma()));
    context.relative(numerical, analytic, 2e-6, "analytic plane optical depth");

    const double majorant = classicMajorant(field, field.activeDomain, external.direction);
    Random rng(8871);
    int escaped = 0;
    constexpr int flightSamples = 5000;
    for (int i = 0; i < flightSamples; ++i) {
        if (!sampleClassicNullTracking(kernel, majorant, rng).collided) ++escaped;
    }
    const double expectedEscape = std::exp(-integrateHazard(
        kernel, kernel.currentAge(), kernel.maximumAgeInDomain()).value);
    context.near(static_cast<double>(escaped) / flightSamples, expectedEscape, 0.025,
                 "null tracking escape frequency");
}

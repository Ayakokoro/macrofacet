#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/BeckmannVisibleSampler.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

namespace {

void testPbrtBeckmannVisibleSampler(TestContext& context) {
    using namespace mf;
    constexpr double alphaX = 0.45;
    constexpr double alphaY = 0.95;
    constexpr int integrationSamples = 24000;
    constexpr int randomSamples = 12000;
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    const Vector3 directions[] = {
        normalizedOrThrow(Vector3(0.75, -0.4, -0.5)),
        normalizedOrThrow(Vector3(-0.6, 0.7, 0.38))
    };
    Random rng(202311);
    for (const Vector3& w : directions) {
        Vector3 expectedMoment = Vector3::Zero();
        double pdfMass = 0.0;
        for (int i = 0; i < integrationSamples; ++i) {
            const double z = 1.0 - 2.0 * (i + 0.5) / integrationSamples;
            const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden * i;
            const Vector3 n(radius * std::cos(phi), radius * std::sin(phi), z);
            const double pdf = pbrtBeckmannVisiblePdf(n, w, alphaX, alphaY);
            pdfMass += pdf;
            expectedMoment += pdf * n;
        }
        pdfMass *= 4.0 * kPi / integrationSamples;
        expectedMoment *= 4.0 * kPi / integrationSamples;
        context.near(pdfMass, 1.0, 0.015, "PBRT Beckmann proposal PDF normalization");

        Vector3 sampledMoment = Vector3::Zero();
        int invalidSamples = 0;
        for (int i = 0; i < randomSamples; ++i) {
            const Vector3 n = samplePbrtBeckmannVisible(w, alphaX, alphaY, rng);
            const double pdf = pbrtBeckmannVisiblePdf(n, w, alphaX, alphaY);
            if (!(n.allFinite() && n.z() * w.z() < 0.0 && -w.dot(n) > 0.0 && pdf > 0.0)) {
                ++invalidSamples;
            }
            sampledMoment += n;
        }
        sampledMoment /= randomSamples;
        context.require(invalidSamples == 0, "PBRT Beckmann sample lies in its PDF support");
        context.require((sampledMoment - expectedMoment).norm() < 0.035,
                        "PBRT Beckmann sampled normal matches the proposal PDF");
    }
    for (const Vector3& w : {Vector3::UnitZ().eval(), Vector3::UnitX().eval(),
                             normalizedOrThrow(Vector3(1.0, 0.2, -1e-5))}) {
        int invalidSamples = 0;
        for (int i = 0; i < 512; ++i) {
            const Vector3 n = samplePbrtBeckmannVisible(w, alphaX, alphaY, rng);
            if (!(n.allFinite() && -w.dot(n) > 0.0 &&
                  pbrtBeckmannVisiblePdf(n, w, alphaX, alphaY) > 0.0)) {
                ++invalidSamples;
            }
        }
        context.require(invalidSamples == 0, "PBRT Beckmann normal and grazing support");
    }
}

void testLocalBeckmannVisibleSampler(TestContext& context) {
    using namespace mf;
    const Vector3 normal = normalizedOrThrow(Vector3(0.55, -0.6, 0.58));
    const Vector3 tangentX = normalizedOrThrow(Vector3::UnitX() - normal.x() * normal);
    const Vector3 tangentY = normal.cross(tangentX);
    const Matrix3 covariance =
        0.07 * (tangentX * tangentX.transpose()) +
        0.18 * (tangentY * tangentY.transpose()) +
        0.06 * (tangentX * tangentY.transpose() + tangentY * tangentX.transpose()) +
        0.04 * (normal * normal.transpose());
    const LocalBeckmannVisibleSampler proposal(2.0 * normal, covariance);
    Random rng(29178);
    constexpr int sampleCount = 20000;
    Vector2 slopeMean = Vector2::Zero();
    Matrix2 slopeSecond = Matrix2::Zero();
    int invalidSamples = 0;
    for (int i = 0; i < sampleCount; ++i) {
        const Vector3 n = proposal.sample(-normal, rng);
        const double cosine = normal.dot(n);
        if (!(n.allFinite() && cosine > 0.0 && proposal.pdf(n, -normal) > 0.0)) {
            ++invalidSamples;
            continue;
        }
        const Vector2 slope(-tangentX.dot(n) / cosine, -tangentY.dot(n) / cosine);
        slopeMean += slope;
        slopeSecond += slope * slope.transpose();
    }
    context.require(invalidSamples == 0, "local Beckmann samples use the curved tangent plane");
    slopeMean /= sampleCount;
    slopeSecond /= sampleCount;
    const Matrix2 expected = (Matrix2() << 0.0175, 0.015, 0.015, 0.045).finished();
    context.require(slopeMean.norm() < 0.015 && (slopeSecond - expected).norm() < 0.012,
                    "local Beckmann slopes retain projected anisotropic covariance");

    const Vector3 w = normalizedOrThrow(-0.7 * normal + 0.5 * tangentX);
    Vector3 expectedMoment = Vector3::Zero();
    double pdfMass = 0.0;
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    constexpr int integrationSamples = 24000;
    for (int i = 0; i < integrationSamples; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / integrationSamples;
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = golden * i;
        const Vector3 n(radius * std::cos(phi), radius * std::sin(phi), z);
        const double pdf = proposal.pdf(n, w);
        pdfMass += pdf;
        expectedMoment += pdf * n;
    }
    pdfMass *= 4.0 * kPi / integrationSamples;
    expectedMoment *= 4.0 * kPi / integrationSamples;
    context.near(pdfMass, 1.0, 0.015, "rotated Beckmann proposal PDF normalization");
    Vector3 sampledMoment = Vector3::Zero();
    for (int i = 0; i < sampleCount; ++i) sampledMoment += proposal.sample(w, rng);
    sampledMoment /= sampleCount;
    context.require((sampledMoment - expectedMoment).norm() < 0.035,
                    "rotated Beckmann samples match their world-space PDF");
}

} // namespace

void testMacrofacetBaseline(TestContext& context) {
    using namespace mf;
    testPbrtBeckmannVisibleSampler(context);
    testLocalBeckmannVisibleSampler(context);
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

    MaterialConfig unitMaterial;
    unitMaterial.conductor.forceUnitFresnel = true;
    ConductorPhase uniformPhase(field, unitMaterial, Point3::Zero(), 0.0);
    ConductorPhase mixedPhase(field, unitMaterial, Point3::Zero(), 0.5);
    ConductorPhase targetPhase(field, unitMaterial, Point3::Zero(), 0.0, true);
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
    GPSSField curvedField = field;
    curvedField.mean = std::make_shared<SphereMean>(Point3::Zero(), 1.0);
    ConductorPhase curvedPhase(curvedField, unitMaterial, Point3::UnitX(), 0.9);
    Random curvedRng(19491);
    Vector3 curvedNormalMean = Vector3::Zero();
    for (int i = 0; i < 256; ++i) {
        const PhaseSample sample = curvedPhase.samplePhase(-Vector3::UnitX(), curvedRng);
        curvedNormalMean += sample.sampledNormal;
        context.relative(sample.samplingPdf,
                         curvedPhase.evaluateSamplingPdf(-Vector3::UnitX(), sample.direction),
                         1e-10, "curved phase sample and PDF agree");
    }
    curvedNormalMean /= 256.0;
    context.require(curvedNormalMean.x() > 0.65 && std::abs(curvedNormalMean.z()) < 0.15,
                    "curved phase Beckmann proposal follows the local sphere normal");
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
    ClassicFlightKernel kernel(field, unitMaterial, external);
    const double end = std::min(0.3, kernel.maximumAgeInDomain());
    const double numerical = integrateHazard(kernel, 0.0, end).value;
    const double area = ndf.projectedArea(external.direction).value;
    const double d0 = 0.2;
    const double wz = external.direction.z();
    const double analytic = area / wz *
        (normalLogCdf((d0 + wz * end) / field.kernel.sigma()) -
         normalLogCdf(d0 / field.kernel.sigma()));
    context.relative(numerical, analytic, 2e-6, "analytic plane optical depth");

    const double majorant = classicMajorant(field, unitMaterial, field.activeDomain, external.direction);
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

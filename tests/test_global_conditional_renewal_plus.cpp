#include "TestHarness.h"
#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <limits>

namespace {
class UnitDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 1.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override { return {1.0,1.0,true}; }
};
} // namespace

void testGlobalConditionalRenewalPlus(TestContext& context) {
    using namespace mf;
    // A curved mean, anisotropic kernel, and nonzero birth position expose any
    // accidental tangent-plane approximation or change of coordinates/kernel.
    GPSSField field{std::make_shared<SphereMean>(Point3(0.1,-0.2,0.05),0.3),
        CovarianceKernel::fromGradientCovariance(CovarianceKernelType::SquaredExponential,
            0.1,Vector3(0.04,0.1,0.16).asDiagonal()),
        {Point3::Constant(-1),Point3::Constant(1)}};
    const GlobalConditionalRenewalPlusMedium oneCurvedSegment(field,0.5);
    const ConditionalMedium global(field);
    Random initialized(311), referenceInitialized(311);
    const Point3 entry(0.42,-0.1,0.15);
    const Vector3 direction=normalizedOrThrow(Vector3(-0.7,0.2,0.1));
    const auto initializedBirth=oneCurvedSegment.startExternal(entry,direction,initialized);
    const auto referenceBirth=global.startExternal(entry,direction,referenceInitialized);
    context.near(initializedBirth.birthValue,referenceBirth.birthValue,0,
        "renewal+ and global conditional sample the same exterior value");
    context.near((initializedBirth.birthGradient-referenceBirth.birthGradient).norm(),0,0,
        "renewal+ and global conditional sample the same full exterior gradient");
    context.require(initialized.integer()==referenceInitialized.integer(),
        "exterior initialization consumes the same random stream");
    for (double value : {0.0,0.03}) {
        auto birth=startExternalFlight(entry,direction);
        birth.birthValue=value;
        birth.birthGradient=Vector3(-0.5,0.1,0.1);
        const auto flight=global.beginFlight(birth);
        for (int trial=0;trial<128;++trial) {
            Random a(901+trial), b(901+trial);
            std::uint64_t segments=0;
            const auto renewed=oneCurvedSegment.sample(birth,a,nullptr,0.45,&segments);
            const auto original=global.sample(flight,b,nullptr,0.45);
            context.require(renewed.collided==original.collided && segments==1,
                "one curved renewal segment has exactly the global collision decision");
            context.near(renewed.age,original.age,0,
                "one curved renewal segment has exactly the global distance");
            if (original.collided) {
                const Vector3 gradient=sampleCollisionGradient(flight,original.age,b);
                context.near((renewed.gradient-gradient).norm(),0,0,
                    "one curved renewal segment has exactly the global collision gradient");
            }
            context.require(a.integer()==b.integer(),
                "one segment consumes exactly the global random stream including escape");
        }
    }
    for (double distance : {0.0,-0.01,std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected=false;
        try { (void)GlobalConditionalRenewalPlusMedium(field,distance); }
        catch(const std::invalid_argument&) { rejected=true; }
        context.require(rejected,"renewal distance must be finite and positive");
    }

    field.mean=std::make_shared<PlaneMean>(Vector3::UnitZ(),0);
    field.kernel=CovarianceKernel::fromCorrelationLengths(0.1,Vector3::Constant(0.1));
    auto birth=startExternalFlight(Point3(0,0,0.02),Vector3(0.8,0,-0.6));
    birth.birthValue=0.03; birth.birthGradient=Vector3(0.5,0,0.5);
    const ConditionalFlightKernel posterior(field,birth);
    const double t=0.12;
    const auto m=posterior.scalarStatistics(t);
    const double sd=std::sqrt(m.varianceF), z=m.meanF/sd;
    const double mills=normalPdfOverCdf(z);
    const double expectedF=m.meanF+sd*mills;
    const double varianceF=m.varianceF*(1-z*mills-mills*mills);
    const double u=100*t*t;
    const double regression=100*t*u*std::exp(-u)/(1-(1+u)*std::exp(-u));
    const double expectedK=m.meanK+regression*sd*mills;
    const double varianceK=m.varianceAtZero+regression*regression*varianceF;
    constexpr int draws=24000;
    Random endpoints(80531);
    double sumF=0,sumK=0,sumFK=0;
    bool valid=true;
    for (int i=0;i<draws;++i) {
        const auto endpoint=posterior.samplePositiveEndpoint(t,endpoints);
        const double k=birth.direction.dot(endpoint.birthGradient);
        valid=valid && endpoint.birthValue>0 && endpoint.birthGradient.allFinite();
        sumF+=endpoint.birthValue; sumK+=k;
        sumFK+=(endpoint.birthValue-expectedF)*(k-expectedK);
    }
    context.require(valid,"renewal endpoint has a positive value and finite full gradient");
    context.near(sumF/draws,expectedF,6*std::sqrt(varianceF/draws),
        "positive endpoint value has the truncated posterior mean");
    context.near(sumK/draws,expectedK,6*std::sqrt(varianceK/draws),
        "endpoint gradient retains its correlation with the sampled value");
    context.near(sumFK/draws,regression*varianceF,8*std::sqrt(varianceF*varianceK/draws),
        "joint positive endpoint covariance matches Gaussian conditioning");

    // Independent quadrature checks the collision law, including escape mass.
    const GlobalConditionalRenewalPlusMedium oneSegment(field,0.2);
    constexpr int bins=4000;
    double integral=0;
    for (int i=0;i<bins;++i) integral+=posterior.evaluate((i+0.5)*0.2/bins).hazard.value*0.2/bins;
    const double expectedSurvival=std::exp(-integral);
    Random tracking(29375);
    constexpr int trials=8000;
    int escaped=0, hits=0;
    bool gradients=true;
    for (int i=0;i<trials;++i) {
        const auto sample=oneSegment.sample(birth,tracking,nullptr,0.2);
        if (!sample.collided) ++escaped;
        else { ++hits; gradients=gradients && birth.direction.dot(sample.gradient)<0; }
    }
    context.require(expectedSurvival>0.01 && expectedSurvival<0.99 && hits>0,
        "renewal sampling test exercises collisions and escapes");
    context.near(static_cast<double>(escaped)/trials,expectedSurvival,
        6*std::sqrt(expectedSurvival*(1-expectedSurvival)/trials),
        "renewal delta tracking agrees with independent hazard integration");
    context.require(gradients,"renewal collision gradients have negative incident flux");

    // Transverse gradient shifts cannot affect scalar events on an isotropic
    // straight ray, but their memory must reach later collision gradients.
    // This distinguishes full-gradient renewal from direction-only renewal.
    constexpr double distance=0.03, ell=0.1;
    const GlobalConditionalRenewalPlusMedium stepped(field,distance);
    birth=startExternalFlight(Point3::Zero(),Vector3::UnitX());
    birth.birthValue=0.03; birth.birthGradient=Vector3(0.1,0,0.5);
    const Vector3 transverseShift(0,0.7,-0.4);
    auto shifted=birth; shifted.birthGradient+=transverseShift;
    int laterHits=0;
    for (int trial=0;trial<128;++trial) {
        Random a(31415+trial), b(31415+trial);
        std::uint64_t segments=0, shiftedSegments=0;
        const auto original=stepped.sample(birth,a,nullptr,0.3,&segments);
        const auto changed=stepped.sample(shifted,b,nullptr,0.3,&shiftedSegments);
        context.require(original.collided==changed.collided && segments==shiftedSegments,
            "transverse birth gradients do not change the scalar event or renewal count");
        context.near(original.age,changed.age,0,"transverse birth gradients do not change collision distance");
        if (original.collided && segments>1) {
            ++laterHits;
            const double last=original.age-(segments-1)*distance;
            const double attenuation=std::exp(-0.5*((segments-1)*distance*distance+last*last)/(ell*ell));
            context.near((changed.gradient-original.gradient-attenuation*transverseShift).norm(),0,1e-10,
                "both transverse gradient components survive every renewal with the SE correlation");
        }
    }
    context.require(laterHits>0,"full-gradient memory check reaches collisions after a renewal");

    field.mean=std::make_shared<ConstantMean>(10);
    const GlobalConditionalRenewalPlusMedium empty(field,0.01);
    birth=startExternalFlight(Point3::Zero(),Vector3::UnitX());
    birth.birthValue=10; birth.birthGradient=Vector3::Zero();
    Random stepRng(1238);
    std::uint64_t segments=0;
    const auto escapedStep=empty.sample(birth,stepRng,nullptr,0.035,&segments);
    context.require(!escapedStep.collided && segments==4,"fixed renewal distances include a shorter final segment");
    context.near(escapedStep.age,0.035,0,"renewal escape reports the accumulated world distance");
    segments=0;
    const auto domainEscape=GlobalConditionalRenewalPlusMedium(field,0.3).sample(birth,stepRng,nullptr,-1,&segments);
    context.require(!domainEscape.collided && segments==4,"renewals clip the final segment to the global domain");
    context.near(domainEscape.age,1,0,"domain escape uses the original flight distance");

    ExperimentConfig render;
    render.field=field; render.transportMode="global_conditional_renewal_plus";
    render.field.mean=std::make_shared<PlaneMean>(Vector3::UnitZ(),0);
    render.field.activeDomain={Point3(-0.1,-0.1,-0.03),Point3(0.1,0.1,0.03)};
    render.render.width=3; render.render.height=3; render.render.samplesPerPixel=4;
    render.render.cameraPosition=Point3(0,0,0.2); render.render.cameraTarget=Point3::Zero();
    render.render.threadCount=1; render.render.environment="unit_white";
    render.render.rouletteStartDepth=100; render.render.safetyDepthCap=128;
    render.material.conductor.forceUnitFresnel=true;
    const auto serial=renderAnalyticScene(render);
    render.render.threadCount=3;
    const auto parallel=renderAnalyticScene(render);
    context.require(serial.statistics.numericalFailures==0 && serial.statistics.safetyCapTerminations==0 &&
        serial.statistics.conditionalRenewalSegments>serial.statistics.conditionalFlights,
        "renewal renderer executes multiple segments per flight without numerical failures");
    for (std::size_t i=0;i<serial.pixels.size();++i) {
        context.near((serial.pixels[i]-Spectrum::Ones()).norm(),0,1e-14,"renewal+ white furnace preserves energy");
        context.require(serial.pixels[i]==parallel.pixels[i],"renewal+ rendering is deterministic across thread counts");
    }
    context.require(serial.statistics.conditionalRenewalSegments==parallel.statistics.conditionalRenewalSegments,
        "renewal segment diagnostics merge across rendering threads");

    render.conditionalRenewalDistance=1.0; // Longer than any flight in this box.
    render.render.environment="directional_gradient";
    const auto oneSegmentImage=renderAnalyticScene(render);
    render.transportMode="global_conditional";
    render.material.gpModel=GpModel::GlobalPointwise;
    render.mediumDensity=std::make_shared<UnitDensity>();
    render.densityMajorantGrid=std::make_shared<DensityMajorantGrid>(
        *render.mediumDensity,render.field.activeDomain,4);
    const auto globalImage=renderAnalyticScene(render);
    context.require(oneSegmentImage.pixels==globalImage.pixels &&
        oneSegmentImage.statistics.realCollisions==globalImage.statistics.realCollisions &&
        oneSegmentImage.statistics.tracking.candidates==globalImage.statistics.tracking.candidates,
        "single-segment renewal rendering exactly matches global conditional including camera initialization");
}

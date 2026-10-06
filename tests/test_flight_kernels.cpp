#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/transport/CollisionGradientSampler.h"

namespace {

// A continuous mean with the same derivative jump as a trilinear cell face.
// Its zero after the turn is only 2e-7 from the birth observation.
class TurningMean final : public mf::MeanField {
public:
    static constexpr double turn=1e-7;
    const char* typeName() const override { return "turning_test"; }
    mf::MeanJet evaluate(const mf::Point3& x) const override {
        return {x.x()<=turn ? x.x() : 2.0*turn-x.x(),
                x.x()<=turn ? mf::Vector3::UnitX().eval() : (-mf::Vector3::UnitX()).eval()};
    }
    mf::BoundsSummary bounds(const mf::Bounds3& box) const override {
        return {std::min(evaluate(box.minimum).value,evaluate(box.maximum).value),1.0,true};
    }
    mf::MeanRayBounds rayBounds(const mf::Point3& origin, const mf::Vector3& w,
                                double begin, double end) const override {
        const auto first=evaluate(origin+begin*w), last=evaluate(origin+end*w);
        mf::MeanRayBounds b;
        b.minimumValue=std::min(first.value,last.value);
        b.maximumValue=std::max(first.value,last.value);
        b.minimumDerivative=std::min(first.gradient.dot(w),last.gradient.dot(w));
        b.maximumDerivative=std::max(first.gradient.dot(w),last.gradient.dot(w));
        b.beginDerivative=first.gradient.dot(w);
        b.maximumSecondDerivative=0.0;
        if (w.x()!=0.0) {
            const double knot=(turn-origin.x())/w.x();
            if (knot>=begin && knot<=end) {
                b.maximumValue=turn;
                b.minimumDerivative=-std::abs(w.x());
                b.maximumDerivative=std::abs(w.x());
                b.maximumSecondDerivative=std::numeric_limits<double>::infinity();
            }
        }
        b.certified=true;
        return b;
    }
    void appendRayBreakpoints(const mf::Point3& origin, const mf::Vector3& w,
        double begin, double end, std::vector<double>& knots) const override {
        if (w.x()!=0.0) {
            const double knot=(turn-origin.x())/w.x();
            if (knot>begin && knot<end) knots.push_back(knot);
        }
    }
};

} // namespace

void testFlightKernels(TestContext& context) {
    using namespace mf;
    GPSSField field = buildDefaultField();
    MaterialConfig material;
    const Vector3 direction = normalizedOrThrow(Vector3(0.9, 0.0, 0.435889894));
    const FlightState state = startExternalFlight(Point3::Zero(), direction);
    ClassicFlightKernel kernel(field, material, state);
    for (double age : {0.01, 0.03, 0.08, 0.15, 0.25}) {
        const auto hazard = kernel.evaluate(age).hazard;
        context.require(hazard.value >= 0.0 && std::isfinite(hazard.value),
                        "Classic hazard is finite and nonnegative");
        const auto gradient = kernel.hitStatistics(age).collisionGradient;
        context.require(gradient.mean.allFinite() && gradient.covariance.allFinite(),
                        "Classic collision gradient is finite");
    }
    // A long SE separation must recover the global one-point classic model.
    field = {std::make_shared<PlaneMean>(Vector3::UnitZ(), 0.0),
        SquaredExponentialKernel::fromCorrelationLengths(0.1,
            Vector3::Constant(0.1)),
        {Point3(-2.0, -2.0, -2.0), Point3(2.0, 2.0, 2.0)}};
    FlightState surface = startExternalFlight(Point3::Zero(), Vector3::UnitX());
    surface.birthGradient = Vector3(0.2, 0.0, 1.0);
    ConditionalFlightKernel conditioned(field, surface);
    {
        GPSSField maternField = field;
        maternField.kernel = CovarianceKernel::fromGradientCovariance(
            CovarianceKernelType::Matern52, field.kernel.sigma(),
            field.kernel.gradientCovarianceAtZero());
        bool rejected = false;
        try {
            (void)ConditionalFlightKernel(maternField, surface);
        } catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("global_conditional") !=
                       std::string::npos;
        }
        context.require(rejected,
                        "non-SE conditional transport is rejected instead of using SE formulas");
    }
    material.gpModel = GpModel::GlobalPointwise;
    ClassicFlightKernel pointwise(field, material, surface);
    const double age = 0.8;
    context.relative(conditioned.evaluate(age).hazard.value,
        pointwise.evaluate(age).hazard.value, 1e-8,
        "conditional hazard approaches global classic at long separation");
    const auto conditionedG = conditioned.hitStatistics(age).collisionGradient;
    const auto prior = field.pointPrior(Point3(age, 0.0, 0.0));
    context.near((conditionedG.mean - prior.meanG).norm(), 0.0, 1e-8,
        "conditional collision gradient approaches prior mean");
    context.near((conditionedG.covariance - prior.covarianceG).norm(), 0.0, 1e-8,
        "conditional collision gradient approaches prior covariance");
    Random rng(345);
    const FlightSample flight = sampleConditionalDeltaTracking(conditioned, rng,
        nullptr, 0.4);
    context.require(flight.age >= 0.0 && flight.age <= 0.4,
                    "conditional delta sample remains in requested interval");
    if (flight.collided) {
        const Vector3 gradient = sampleCollisionGradient(conditioned, flight.age, rng);
        context.require(surface.direction.dot(gradient) < 0.0,
                        "conditional collision uses negative-flux gradient");
    }
    {
        const ConditionalMedium medium(field);
        DdaTrackingDiagnostics proof;
        auto cursor=medium.sampleRay(conditioned,0.4,&proof);
        context.require(proof.boundIntervals==0,
                        "conditional cursor does not prove the far ray at construction");
        double covered=0.0;
        int count=0;
        while (const auto segment=cursor.next()) {
            context.near(segment->beginAge,covered,1e-15,
                         "conditional segments have no gaps or overlaps");
            context.require(segment->endAge>segment->beginAge &&
                            std::isfinite(segment->majorant),
                            "conditional segment has a finite bound and advances");
            for (int j=0;j<4;++j) {
                const double t=segment->beginAge+(segment->endAge-segment->beginAge)*
                    (j+0.5)/4.0;
                context.require(conditioned.evaluate(t).hazard.value<=segment->majorant,
                                "conditional segment covers interior extinction probes");
            }
            covered=segment->endAge;
            ++count;
        }
        context.near(covered,0.4,1e-15,"conditional cursor covers requested flight");
        context.require(count>1 && proof.boundIntervals>0,
                        "conditional cursor computes far bounds when visited");
    }

    // Independent 4x4 Gaussian conditioning reference at a non-small separation.
    {
        const double t=0.07;
        const auto jet=field.kernel.evaluate(surface.birthPosition+t*surface.direction,
                                             surface.birthPosition);
        Eigen::Matrix4d priorCov=Eigen::Matrix4d::Zero(), cross;
        priorCov(0,0)=0.01;
        priorCov.bottomRightCorner<3,3>()=0.01*field.kernel.precision();
        cross(0,0)=jet.valueValue;
        cross.block<1,3>(0,1)=jet.valueXGradientY.transpose();
        cross.block<3,1>(1,0)=jet.gradientXValueY;
        cross.bottomRightCorner<3,3>()=jet.gradientXGradientY;
        Eigen::Vector4d birthMean, targetMean, observed;
        birthMean << 0.0,0.0,0.0,1.0;
        targetMean=birthMean;
        observed << surface.birthValue,surface.birthGradient;
        const Eigen::Vector4d posteriorMean=targetMean+cross*priorCov.inverse()*(observed-birthMean);
        const Eigen::Matrix4d posteriorCov=priorCov-cross*priorCov.inverse()*cross.transpose();
        const Vector3 gf=posteriorCov.block<3,1>(1,0);
        const Vector3 gm=posteriorMean.tail<3>()-gf*posteriorMean[0]/posteriorCov(0,0);
        const Matrix3 gc=posteriorCov.bottomRightCorner<3,3>()-gf*gf.transpose()/posteriorCov(0,0);
        const auto actual=conditioned.hitStatistics(t).collisionGradient;
        context.near((actual.mean-gm).norm(),0.0,1e-11,"conditional mean matches matrix Schur complement");
        context.near((actual.covariance-gc).norm(),0.0,1e-11,"stable covariance matches matrix Schur complement");
    }
    for (double t : {1e-5,1e-7,1e-9}) {
        const auto scalar=conditioned.scalarStatistics(t);
        context.near(scalar.varianceF/(0.005*10000.0*std::pow(t,4)),1.0,1e-7,
                     "near-birth value variance retains t^4 limit");
        context.near(scalar.varianceAtZero/(0.01*1000000.0*std::pow(t,4)/6.0),1.0,1e-7,
                     "near-birth directional variance retains t^4 limit");
    }
    // Search logarithmic ages too: grazing peaks can be much narrower than a
    // voxel or the old 17-point scanning step.
    for (double slope : {0.2,0.002,0.00001}) {
        FlightState grazing=surface;
        grazing.birthGradient.x()=slope;
        for (double exteriorValue : {0.0,0.03}) {
            grazing.birthValue=exteriorValue;
            const ConditionalFlightKernel testKernel(field,grazing);
            const auto bound=testKernel.twoSegmentMajorants(0.3);
            context.require(bound.split>0.0 && bound.split<=0.3,
                            "conditional majorant has a finite birth split");
            for (int i=0; i<=300; ++i) {
                const double t=0.3*std::pow(10.0,-12.0+12.0*i/300.0);
                const double maximum=t<bound.split ? bound.nearMaximum : bound.farMaximum;
                context.require(testKernel.evaluate(t).hazard.value<=maximum,
                                "two analytic constants cover logarithmic conditional hazard probes");
            }
        }
    }
    // Check the finite-interval bound directly across both signs of the
    // conditional value and slope means, rather than only its far maximum.
    for (double slope : {0.2,0.002,0.00001}) {
        for (double exteriorValue : {0.0,0.03}) {
            FlightState observed=surface;
            observed.birthGradient.x()=slope;
            observed.birthValue=exteriorValue;
            const ConditionalFlightKernel intervalKernel(field,observed);
            for (double lo : {0.002,0.01,0.05,0.2}) {
                const double hi=1.5*lo;
                const double majorant=intervalKernel.intervalMajorant(lo,hi);
                context.require(std::isfinite(majorant) && majorant>=0.0,
                                "conditional interval has a finite nonnegative bound");
                for (int i=0;i<=64;++i) {
                    const double t=lo+(hi-lo)*i/64.0;
                    context.require(intervalKernel.evaluate(t).hazard.value<=majorant,
                                    "tightened conditional interval covers dense hazard probes");
                }
            }
        }
    }
    // Empirical survival must match independent integration of the actual
    // hazard, including a resumed flight whose birth age remains unchanged.
    for (double begin : {0.0,0.04}) {
        FlightState resumed=surface;
        resumed.age=begin;
        const ConditionalFlightKernel testKernel(field,resumed);
        const double end=0.25, step=(end-begin)/2048.0;
        double opticalDepth=0.0;
        for (int i=0; i<2048; ++i)
            opticalDepth+=step*testKernel.evaluate(begin+(i+0.5)*step).hazard.value;
        const double expected=std::exp(-opticalDepth);
        int escapes=0;
        Random sampleRng(957+(begin>0.0 ? 1 : 0));
        DdaTrackingDiagnostics tracking;
        constexpr int trials=2000;
        for (int i=0; i<trials; ++i)
            if (!sampleConditionalDeltaTracking(testKernel,sampleRng,&tracking,end).collided) ++escapes;
        context.near(static_cast<double>(escapes)/trials,expected,
                     6.0*std::sqrt(expected*(1.0-expected)/trials)+0.002,
                     "two-segment delta tracking agrees with integrated survival");
        context.require(tracking.candidates==tracking.nearCandidates+tracking.farCandidates,
                        "conditional candidate counters cover both segments");
    }
    {
        GPSSField tinyVariance{std::make_shared<PlaneMean>(-Vector3::UnitX(),0.0),
            SquaredExponentialKernel::fromCorrelationLengths(1e-10,Vector3::Ones()),
            {Point3(-0.1,-1,-1),Point3(2.2,1,1)}};
        FlightState resumed=startExternalFlight(Point3::Zero(),Vector3::UnitX());
        resumed.birthGradient=Vector3::UnitX();
        resumed.age=2.0;
        const ConditionalFlightKernel smallSteps(tinyVariance,resumed);
        const auto m=smallSteps.scalarStatistics(2.0);
        context.relative(smallSteps.evaluate(2.0).hazard.value/(-m.meanF*(-m.meanAtZero)/m.varianceF),
                         1.0,1e-12,"conditional far-tail hazard retains the inverse Mills factor");
        Random tinyRng(91024);
        DdaTrackingDiagnostics tinyTracking;
        const auto hit=sampleConditionalDeltaTracking(smallSteps,tinyRng,&tinyTracking,2.1);
        context.require(hit.collided && hit.age==2.0 && tinyTracking.roundedCandidateSteps>0,
                        "a sub-ulp exponential candidate is retained and accepted without a minimum step");
    }
    {
        GPSSField turned{std::make_shared<TurningMean>(),
            SquaredExponentialKernel::fromCorrelationLengths(0.001,Vector3::Constant(0.0044)),
            {Point3(-0.001,-1,-1),Point3(0.001,1,1)}};
        FlightState birth=startExternalFlight(Point3::Zero(),Vector3::UnitX());
        birth.birthGradient=Vector3::UnitX();
        const ConditionalFlightKernel narrowPeak(turned,birth);
        Random peakRng(4247);
        DdaTrackingDiagnostics peakTracking;
        for (int i=0; i<32; ++i) {
            const auto hit=sampleConditionalDeltaTracking(narrowPeak,peakRng,&peakTracking);
            context.require(hit.collided && std::abs(hit.age-2.0*TurningMean::turn)<1e-9,
                            "cell-face derivative jump resolves its narrow first-collision peak");
        }
        context.require(peakTracking.adaptiveMajorantFlights==32 && peakTracking.candidates<32000,
                        "certified interval thinning avoids billions of global null events");
    }
    {
        FlightState grazing=surface;
        grazing.birthGradient.x()=1e-5;
        const ConditionalFlightKernel accelerated(field,grazing);
        const double end=0.25, logBegin=std::log(1e-12);
        const double step=(std::log(end)-logBegin)/4096.0;
        double depth=0.0;
        for (int i=0; i<4096; ++i) {
            const double t=std::exp(logBegin+(i+0.5)*step);
            depth+=accelerated.evaluate(t).hazard.value*t*step;
        }
        const double expected=std::exp(-depth);
        Random acceleratedRng(5819);
        DdaTrackingDiagnostics tracking;
        constexpr int trials=1000;
        int escapes=0;
        for (int i=0; i<trials; ++i)
            if (!sampleConditionalDeltaTracking(accelerated,acceleratedRng,&tracking,end).collided) ++escapes;
        context.require(tracking.boundIntervals>0,"grazing survival test exercises lazy interval bounds");
        context.near(static_cast<double>(escapes)/trials,expected,
                     6.0*std::sqrt(expected*(1.0-expected)/trials)+0.002,
                     "accelerated tracking agrees with independent log-distance hazard integration");
    }
}

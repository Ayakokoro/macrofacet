#include "TestHarness.h"
#include "macrofacet/gpss/TungstenFunctionSpaceSampler.h"
#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

namespace {
using namespace mf;
template<class F> bool rejects(F f) { try { f(); } catch(const std::exception&) { return true; } return false; }
TungstenFunctionSpaceBatch prescribed(std::initializer_list<double> values) {
    TungstenFunctionSpaceBatch b;
    b.values=Eigen::Map<const Eigen::VectorXd>(values.begin(),values.size());
    for (int i=0;i<b.values.size();++i) {
        b.times.push_back(0.1+i);
        b.observations.push_back({Point3(0.1+i,0,0),false,Vector3::Zero()});
    }
    b.distance=b.times.back();
    return b;
}
}

void testTungstenFunctionSpace(TestContext& context) {
    using namespace mf;
    // Frozen outputs from the actual local Tungsten functions, built in an
    // independent harness. Replay uniforms so RNG seeding cannot hide a
    // difference in sampling order or batch/gradient behavior.
    const auto fixturePath=std::filesystem::path(__FILE__).parent_path()/"data/tungsten_function_space_reference.json";
    std::ifstream fixtureFile(fixturePath);
    if (!fixtureFile) throw std::runtime_error("missing Tungsten differential fixture: "+fixturePath.string());
    nlohmann::json fixture; fixtureFile>>fixture;
    const auto vec=[](const nlohmann::json& a) { return Vector3(a[0].get<double>(),a[1].get<double>(),a[2].get<double>()); };
    for (const auto& entry:fixture.at("uniforms")) {
        TungstenRandom rng(entry.at("seed").get<std::uint64_t>());
        for (double expected:entry.at("values")) context.near(rng.uniform01(),expected,0,"Tungsten PCG reference uniform");
    }
    int fixtureIndex=0;
    for (const auto& entry:fixture.at("batches")) {
        GPSSField referenceField;
        if (entry.at("plane").get<bool>()) referenceField.mean=std::make_shared<PlaneMean>(Vector3::UnitZ(),-.1);
        else referenceField.mean=std::make_shared<ConstantMean>(0);
        referenceField.kernel=CovarianceKernel::fromCorrelationLengths(.2,Vector3::Constant(.4));
        const TungstenFunctionSpaceSampler referenceSampler(referenceField,
            {entry.at("sample_points").get<int>(),entry.at("step_size").get<double>()});
        std::optional<TungstenSurfaceState> previous;
        if (!entry.at("previous").is_null()) {
            const auto& p=entry.at("previous");
            previous=TungstenSurfaceState{vec(p.at("position")),p.at("value").get<double>(),vec(p.at("gradient"))};
        }
        std::size_t drawIndex=0;
        const auto tape=entry.at("uniforms").get<std::vector<double>>();
        TungstenRandom replay([&] { return tape.at(drawIndex++); });
        const auto result=referenceSampler.sampleBatch({vec(entry.at("origin")),vec(entry.at("direction"))},
            entry.at("near").get<float>(),entry.at("far").get<float>(),previous,
            entry.at("first_intersection").get<bool>(),replay);
        const std::string label="Tungsten source fixture batch "+std::to_string(fixtureIndex++);
        context.require(result.hit==entry.at("hit").get<bool>(),label+" crossing");
        context.require(drawIndex==tape.size(),label+" exact uniform consumption");
        context.near(result.distance,entry.at("distance").get<double>(),2e-7,label+" distance");
        context.near((result.end.gradient-vec(entry.at("gradient"))).norm(),0,2e-5,label+" full gradient");
        const auto expected=entry.at("values").get<std::vector<double>>();
        context.require(result.values.size()==static_cast<Eigen::Index>(expected.size()),label+" observation count");
        for (Eigen::Index j=0;j<result.values.size() && j<static_cast<Eigen::Index>(expected.size());++j) {
            context.near(result.values[j],expected[j],2e-6,label+" field/derivative observation");
            context.near((result.observations[j].position-vec(entry.at("points").at(j))).norm(),0,2e-7,label+" observation position");
        }
    }
    GPSSField field{std::make_shared<ConstantMean>(0),
        CovarianceKernel::fromCorrelationLengths(1.,Vector3::Ones()),
        {Point3::Constant(-4),Point3::Constant(4)}};
    TungstenFunctionSpaceSampler sampler(field,{8,0.1});

    // Independent closed-form conditional moments. The condition is a value
    // and a projected gradient, not the full gradient or H grid observations.
    std::vector<TungstenObservation> observations{
        {Point3::Zero(),false,Vector3::Zero()},{Point3::Zero(),true,Vector3::UnitX()}};
    Eigen::VectorXd values(2); values << .7,-.4;
    std::vector<TungstenObservation> query;
    for (double t : {.1,.4,1.2}) query.push_back({Point3(t,0,0),false,Vector3::Zero()});
    const auto gaussian=sampler.conditional(query,observations,values);
    for (int i=0;i<3;++i) {
        const double t=query[i].position.x(), r=std::exp(-.5*t*t);
        context.near(gaussian.mean[i],r*(.7-.4*t),1e-14,"Tungsten renewal+ conditional mean");
        for (int j=0;j<3;++j) {
            const double s=query[j].position.x();
            context.near(gaussian.covariance(i,j),std::exp(-.5*(t-s)*(t-s))-
                (1+t*s)*std::exp(-.5*(t*t+s*s)),1e-14,"Tungsten renewal+ joint covariance");
        }
    }
    // Scalar and rank-deficient Gaussian draws exercise both decomposition paths.
    int inputs=0;
    TungstenRandom zeros([&] { ++inputs; return 0.; });
    TungstenGaussian singular{Eigen::VectorXd::Constant(3,2),Eigen::MatrixXd::Zero(3,3)};
    context.near((TungstenFunctionSpaceSampler::draw(singular,zeros)-singular.mean).norm(),0,0,
        "Tungsten eigen fallback preserves a singular Gaussian");
    context.require(inputs==4,"odd-dimensional Gaussian discards the second Box-Muller variate");
    const Ray ray{Point3::Zero(),Vector3::UnitX()};
    const auto initial=sampler.sampleBatch(ray,0.f,2.f,std::nullopt,true,zeros);
    context.near(initial.times.front(),.01,1e-15,"Tungsten first point starts at 0.1h");
    context.near(initial.times.back(),.8,1e-15,"Tungsten batch spans N*h");
    context.near(initial.times[1],.01+.79/7,1e-15,"Tungsten interior placement uses N-1");
    context.require(zeros.draws()==4+1+2+8+4,"first batch consumes unused positive start and escaped gradient draws");
    context.near(initial.values.norm(),0,0,"positive start draw does not condition the first batch");

    auto first=prescribed({1.,-1.,-2.,1.});
    auto surface=first;
    TungstenFunctionSpaceSampler::locateIntersection(first,ray,false);
    TungstenFunctionSpaceSampler::locateIntersection(surface,ray,true);
    context.near(first.distance,.6,1e-15,"camera ray detects first interval crossing");
    context.near(surface.distance,2.1+2./3,1e-15,"surface ray skips first interval and resets sign");
    context.near(surface.values[surface.values.size()-1],3.,1e-15,"hit keeps signed secant derivative");
    context.near(surface.end.value,0,3e-16,"hit preserves interpolated field value");
    auto laterBatch=prescribed({1.,-1.});
    TungstenFunctionSpaceSampler::locateIntersection(laterBatch,ray,false);
    context.require(laterBatch.hit,"continuation batches in the same ray do not repeat first-interval skip");
    auto missed=prescribed({1.,-1.});
    TungstenFunctionSpaceSampler::locateIntersection(missed,ray,true);
    context.require(!missed.hit,"near-surface first return is deliberately skipped like Tungsten");

    // No-hit batches must carry the nonzero endpoint observation. A handoff
    // reconstructed from only a surface normal changes the future process.
    TungstenRandom r1(741),r2(741);
    TungstenSurfaceState birth{Point3::Zero(),.8,Vector3(.6,-.2,.4)};
    auto batch=sampler.sampleBatch(ray,0,.8f,birth,true,r1);
    const auto clipped=sampler.sampleBatch(ray,0,.05f,birth,true,r2);
    context.near(clipped.times.back(),double(.05f),1e-15,"short final batch is laid out within remaining interval");
    context.require(batch.end.gradient.allFinite(),"batch endpoint has a full conditional gradient");

    // At tiny noise, crossing/rejection is determined by the analytic plane.
    GPSSField plane{std::make_shared<PlaneMean>(Vector3::UnitZ(),0),
        CovarianceKernel::fromCorrelationLengths(1e-7,Vector3::Ones()),
        {Point3::Constant(-1),Point3::Constant(1)}};
    FunctionSpaceGpisMedium medium(plane,{8,.06});
    TungstenRandom downRng(12),upRng(12);
    const auto down=medium.sample({Point3(0,0,.8),-Vector3::UnitZ()},std::nullopt,downRng);
    const auto up=medium.sample({Point3(0,0,-.8),Vector3::UnitZ()},std::nullopt,upRng);
    context.require(down.status==GpisFlightStatus::Hit && down.surface.gradient.z()>0,
        "Tungsten accepts downcrossing with original full gradient");
    context.require(up.status==GpisFlightStatus::RejectedUpcrossing,"Tungsten rejects upcrossing instead of flipping the field");
    context.near(down.position.z(),0,1e-6,"Tungsten world-space plane hit");
    context.require(down.batches>1,"no-hit batches continue within the same flight");

    // Compare a manually continued multi-bounce path to the two-stage renderer.
    ExperimentConfig config;
    config.field={std::make_shared<ConstantMean>(0),
        CovarianceKernel::fromCorrelationLengths(.2,Vector3::Constant(.4)),
        {Point3::Constant(-1),Point3::Constant(1)}};
    config.transportMode="hybrid_first_hit";
    config.firstHit.method=kTungstenFunctionSpaceMethod;
    config.continuationMethod=kTungstenFunctionSpaceMethod;
    config.firstHit.tungsten={8,.12}; config.continuationTungsten={8,.12};
    config.material.ndfFamily=NdfFamily::GeneralizedGaussian;
    config.material.conductor.forceUnitFresnel=true;
    config.render.environment="unit_white";
    config.render.rouletteStartDepth=100; config.render.safetyDepthCap=12;
    const FunctionSpaceGpisMedium pathMedium(config.field,{8,.12});
    int multiBounce=0,rejected=0;
    for (int seed=0;seed<80;++seed) {
        config.render.rouletteStartDepth=seed<40 ? 100 : 2;
        Random expectedSeed(seed), actualRng(seed);
        Random event(firstHitStreamSeed(expectedSeed.integer(),0,0,1));
        TungstenRandom rng(event.integer());
        Ray current{Point3(0,0,1.5),-Vector3::UnitZ()};
        const Ray camera=current;
        std::optional<TungstenSurfaceState> previous;
        double expected=0,weight=1; int depth=0;
        for (;depth<config.render.safetyDepthCap;++depth) {
            const auto hit=pathMedium.sample(current,previous,rng);
            if (hit.status!=GpisFlightStatus::Hit) {
                expected=hit.status==GpisFlightStatus::Escape ? weight : 0;
                rejected+=hit.status==GpisFlightStatus::RejectedUpcrossing;
                break;
            }
            previous=hit.surface;
            current={hit.position,normalizedOrThrow(reflectTravelDirection(current.direction,
                normalizedOrThrow(hit.surface.gradient)))};
            if (depth+1>=config.render.rouletteStartDepth) {
                const double probability=std::clamp(weight,.05,.95);
                if (rng.uniform01()>=probability) { ++depth; break; }
                weight/=probability;
            }
        }
        multiBounce+=depth>1;
        RenderStatistics statistics;
        const auto actual=traceCameraPath(camera,config,actualRng,statistics);
        context.near(actual.x(),expected,0,"two-stage GPIS matches explicit continuous RNG multi-bounce path");
        context.require(statistics.realCollisions==static_cast<std::uint64_t>(depth),"GPIS handoff preserves collision count");
        context.require(statistics.gpisInvalidGradients==0,"GPIS path has finite gradients");
    }
    context.require(multiBounce>0 && rejected>0,"multi-bounce comparison exercises reflections and rejected paths");
    config.render.width=5; config.render.height=4; config.render.samplesPerPixel=4;
    config.render.cameraPosition=Point3(0,0,1.5); config.render.cameraTarget=Point3::Zero();
    config.render.threadCount=1;
    const auto serial=renderAnalyticScene(config);
    config.render.threadCount=3;
    const auto parallel=renderAnalyticScene(config);
    for (std::size_t i=0;i<serial.pixels.size();++i)
        context.near((serial.pixels[i]-parallel.pixels[i]).norm(),0,0,"Tungsten rendering is thread deterministic");
    context.require(serial.statistics.gpisBatches==parallel.statistics.gpisBatches &&
        serial.statistics.gpisRejectedUpcrossings==parallel.statistics.gpisRejectedUpcrossings,
        "Tungsten diagnostics merge across workers");
    config.firstHit.method="gpis_function_space";
    Random mixedRng(15); RenderStatistics mixedStats;
    context.require(traceCameraPath({Point3(0,0,1.5),-Vector3::UnitZ()},config,mixedRng,mixedStats).allFinite(),
        "existing first-hit sampler can feed Tungsten continuation");
    context.require(rejects([&] { TungstenFunctionSpaceSampler bad(field,{1,.1}); }),"reject invalid Tungsten point count");
    auto matern=field; matern.kernel=CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::Matern32,1,Vector3::Ones());
    context.require(rejects([&] { TungstenFunctionSpaceSampler bad(matern,{8,.1}); }),"reject unsupported compatibility kernel");
}

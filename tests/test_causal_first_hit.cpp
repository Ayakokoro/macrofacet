#include "TestHarness.h"
#include "macrofacet/learned/CausalFirstHitModel.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace {
nlohmann::json causalFixture() {
    using Json=nlohmann::json;
    Json b=Json::object(),g=Json::object();
    const auto tensor=[](Json& w,const std::string& name,std::vector<int> shape) {
        int count=1; for (int n:shape) count*=n;
        w[name]={{"shape",shape},{"values",std::vector<float>(count,0)}};
    };
    const auto dense=[&](Json& w,const std::string& name,int input,int output) {
        tensor(w,name+".weight",{output,input}); tensor(w,name+".bias",{output});
    };
    dense(b,"initial.0",2,2); dense(b,"initial.2",2,2);
    dense(b,"segment.0",7,2); dense(b,"segment.2",2,2);
    dense(b,"head.0",5,2); dense(b,"head.2",2,1);
    tensor(b,"gru.weight_ih_l0",{6,2}); tensor(b,"gru.weight_hh_l0",{6,2});
    tensor(b,"gru.bias_ih_l0",{6}); tensor(b,"gru.bias_hh_l0",{6});
    dense(g,"net.0",6,2); dense(g,"net.2",2,2); dense(g,"net.4",2,6);
    return {{"format","macrofacet.causal_first_hit"},{"version",1},{"event","unconditioned_first_zero"},
        {"profile_mode","cubic"},{"kernel","squared_exponential"},{"activation_dtype","float32"},
        {"distance_convention","q=distance_from_entry/ell"},{"mean_convention","b=m/sigma"},
        {"gradient_convention","signed_derivative=-initial_sign*sigma/ell*W"},
        {"state_convention","entering_fixed_q_interval_left_limit_gradient"},
        {"model_config",{{"hidden",2},{"embedding",2},{"width",2},{"history",true},
            {"step",.125},{"integration_step",.015625},{"quadrature_order",8}}},
        {"gradient_config",{{"hidden",2},{"width",2},{"size",2},{"kind","mixture"},{"history",true}}},
        {"backbone",b},{"gradient",g}};
}
}

void testCausalFirstHit(TestContext& context) {
    using namespace mf;
    const auto path=std::filesystem::temp_directory_path()/"macrofacet_causal_first_hit_fixture.json";
    { std::ofstream f(path); f<<causalFixture(); }
    const auto model=std::make_shared<FirstHitSEModel>(FirstHitSEModel::load(path));
    std::filesystem::remove(path);
    context.require(model->causal()!=nullptr && model->maximumStep()==.125,"causal model keeps trained grid");
    const RayMeanProfile profile({{0,2.03,{0,0,0,0}}});
    const RayMeanProfile split({{0,.31,{0,0,0,0}},{.31,2.03,{0,0,0,0}}});
    const RayMeanProfile shortProfile({{0,.03,{0,0,0,0}}});
    for (const auto& p:model->query(profile,{0,.125,.126,1.,2.03})) {
        context.near(p.logTransmittance,-p.q/kPi,2e-14,"zero residual reproduces local exponential survival");
        context.near(p.hitBranchProbability[0],.5,1e-14,"symmetric branch posterior");
    }
    bool rejected=false;
    try { model->query(profile,{.2},.0625); } catch (const std::invalid_argument&) { rejected=true; }
    context.require(rejected,"renderer cannot change the recurrent grid");
    std::vector<const RayMeanProfile*> profiles;
    std::vector<CausalFirstHitUniforms> uniforms;
    Random rng(8821);
    for (int i=0;i<517;++i) {
        profiles.push_back(i%3==0 ? &shortProfile : (i%3==1 ? &split : &profile));
        uniforms.push_back(causalFirstHitUniforms(rng));
    }
    CausalFirstHitSession scalar(*model->causal());
    const auto expected=scalar.sample(profiles,uniforms,1e-10);
    context.require(scalar.sample({},{}).empty(),"empty batch does not launch inference");
    for (std::size_t i=0;i<expected.size();++i) {
        const auto& s=expected[i]; const double q=-kPi*std::log(uniforms[i][1]);
        context.require(s.hit==(q<profiles[i]->maximumX()),"batched optical-depth hit classification");
        context.near(s.q,std::min(q,profiles[i]->maximumX()),1e-10,"sample matches analytic exponential inverse");
        if (s.hit) {
            const float scale=std::log1p(std::exp(0.f))+.01f;
            context.near(s.speed,scale*normalQuantile((1+uniforms[i][3])/2),1e-10,"speed uses normalized truncated normal without extra Rice weight");
        }
    }
    GPSSField field{std::make_shared<ConstantMean>(0),
        CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::SquaredExponential,.02,Vector3(.1,.2,.3)),
        {Point3(-1,-1,-1),Point3(1,1,1)}};
    FirstHitSEMedium medium(field,model);
    const auto flight=*medium.beginFlight({Point3(0,0,4),normalizedOrThrow(Vector3(.03,.02,-1))});
    for (int sign:{1,-1}) {
        Random normalRng(73);
        const auto surface=medium.reconstruct(flight,{true,.2,1.7,sign},normalRng);
        context.near(surface.gradient.dot(flight.ray.direction),-sign*flight.sigma/flight.ell*1.7,1e-14,
            "full anisotropic gradient preserves sampled longitudinal speed");
        context.near(surface.normal.norm(),1,1e-14,"sampled gradient becomes a unit normal");
        context.near((surface.position-flight.ray.origin-.2*flight.ell*flight.ray.direction).norm(),0,1e-14,
            "normalized first-hit distance reconstructs world position");
        context.require(surface.initialSign==sign,"normal reconstruction preserves side for continuation");
    }
    ExperimentConfig config; config.transportMode="hybrid_first_hit"; config.field=field;
    config.firstHit.model=model; config.continuationMethod="global_conditional";
    config.material.ndfFamily=NdfFamily::GeneralizedGaussian; config.material.conductor.forceUnitFresnel=true;
    config.render.cameraPosition=Point3(0,0,4); config.render.cameraTarget=Point3::Zero();
    config.render.width=2; config.render.height=2; config.render.samplesPerPixel=4;
    config.render.safetyDepthCap=128; config.render.rouletteStartDepth=128;
    config.render.environment="unit_white"; config.render.writeFirstHitSamples=true; config.render.threadCount=1;
    const auto image=renderAnalyticScene(config);
    context.require(image.statistics.numericalFailures==0 && image.statistics.safetyCapTerminations==0,
        "causal first-hit with conditioned continuation completes all paths");
    for (const auto& pixel:image.pixels) context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"causal hybrid conserves white furnace energy");
    for (const std::string backend:{"torch_cpu","torch_cuda"}) {
        try { CausalFirstHitSession::validateBackend(backend); }
        catch (const std::runtime_error&) { continue; } // Optional LibTorch/CUDA build.
        CausalFirstHitSession session(*model->causal(),backend,3);
        const auto actual=session.sample(profiles,uniforms,1e-10);
        for (std::size_t i=0;i<actual.size();++i) {
            context.require(actual[i].hit==expected[i].hit && actual[i].initialSign==expected[i].initialSign,
                "compacted device slots preserve event and branch");
            context.near(actual[i].q,expected[i].q,1e-10,"batched device distance agrees with scalar");
            context.near(actual[i].speed,expected[i].speed,1e-6,"batched device speed agrees with scalar");
        }
        config.firstHit.backend=backend; config.firstHit.batchSize=7;
        const auto batched=renderAnalyticScene(config);
        for (std::size_t i=0;i<image.firstHitRecords.size();++i) {
            const auto& a=image.firstHitRecords[i].first; const auto& b=batched.firstHitRecords[i].first;
            context.require(a.hit==b.hit && a.initialSign==b.initialSign,"batched render uses the same per-sample streams");
            context.near((a.position-b.position).norm(),0,1e-8,"batched render collision position");
            context.near((a.gradient-b.gradient).norm(),0,1e-6,"batched render full gradient");
        }
        for (const auto& pixel:batched.pixels) context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"batched hybrid white furnace");
    }
}

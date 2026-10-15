#include "TestHarness.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
using Json = nlohmann::json;
class LocalTestDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 8.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override { return {8.0,8.0,true}; }
};
Json constantModel(bool first) {
    Json weights = Json::object();
    const auto tensor = [&](const std::string& name, std::vector<int> shape) {
        int n = 1; for (int x : shape) n *= x;
        weights[name] = {{"shape",shape},{"values",std::vector<float>(n,0)}};
    };
    const auto linear = [&](const std::string& name,int in,int out) {
        tensor(name+".weight",{out,in}); tensor(name+".bias",{out});
    };
    const auto mlp = [&](const std::string& name,int in,int out) {
        linear(name+".0",in,2); linear(name+".2",2,2); linear(name+".4",2,out);
    };
    Json result{{"version",1},{"activation_dtype","float32"},
        {"kernel",{{"type","squared_exponential"},{"parameterization","unit_length"},{"beta",1.0}}}};
    if (first) {
        mlp("initial_encoder",2,2); mlp("dynamics",5,2); mlp("hazard_head",5,1); mlp("mixture_head",5,6);
        result["format"] = "macrofacet.first_hit_se";
        result["event"] = "unconditioned_first_zero"; result["profile_mode"] = "cubic";
        result["branch_order"] = {1,-1}; result["solver"] = "rk4";
        result["distance_convention"] = "q=distance_from_entry/ell";
        result["mean_convention"] = "b=m/sigma";
        result["gradient_convention"] = "signed_derivative=-initial_sign*sigma/ell*W";
        result["model_config"] = {{"architecture","rno"},{"hidden",2},{"width",2},
            {"mixture_width",2},{"components",2},{"maximum_step",0.125},{"sigma_floor",0.01}};
    } else {
        linear("segment_encoder.0",5,2); linear("segment_encoder.2",2,2);
        mlp("initial_encoder",4,2); mlp("hazard_head",4,4); mlp("mixture_head",7,6);
        tensor("gru.weight_ih_l0",{6,2}); tensor("gru.weight_hh_l0",{6,2});
        tensor("gru.bias_ih_l0",{6}); tensor("gru.bias_hh_l0",{6});
        result["format"] = "macrofacet.renewal";
        result["gru_convention"] = "pytorch_rzn_reset_after";
        result["feature_transform"] = "asinh_first_four_log_dx_identity";
        result["initial_transform"] = "asinh_mode_b0_known_z0_known_d0";
        result["checkpoint_sha256"] = "first_hit_render_test_fixture";
        result["mixture_convention"] = "positive_truncated_components_residual_mean";
        result["model_config"] = {{"hidden",2},{"embedding",2},{"hazard_width",2},
            {"mixture_width",2},{"components",2},{"sigma_floor",0.01}};
    }
    result["weights"] = weights; return result;
}
template<class F> bool rejects(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}

void testFirstHitRender(TestContext& context) {
    using namespace mf;
    const auto path = std::filesystem::temp_directory_path()/"macrofacet_first_hit_render_fixture.json";
    { std::ofstream f(path); f << constantModel(true); }
    const auto model = std::make_shared<const FirstHitSEModel>(FirstHitSEModel::load(path));
    std::filesystem::remove(path);
    std::istringstream renewalStream(constantModel(false).dump());
    const auto renewal = std::make_shared<const RenewalHazardModel>(RenewalHazardModel::load(renewalStream));
    GPSSField field{std::make_shared<ConstantMean>(0),
        CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::SquaredExponential,0.1,Vector3::Ones()),
        {Point3(-1,-1,-1),Point3(1,1,1)}};
    FirstHitSEMedium medium(field,model), reference(field,nullptr);
    const Ray camera{Point3(0,0,4),-Vector3::UnitZ()};
    const auto flight = *medium.beginFlight(camera);
    context.near(flight.entryDistance,3,0,"first-hit camera vacuum distance is separate");
    const auto fromEntry = *medium.beginFlight(flight.ray);
    Random a(182), b(182), ga(482), gb(482);
    const auto external = medium.sample(flight,a,ga), internal = medium.sample(fromEntry,b,gb);
    context.near(external.q,internal.q,1e-14,"first-hit event ignores camera-to-entry distance");
    context.near((external.position-internal.position).norm(),0,1e-13,"entry origins reconstruct same collision");
    context.require(!medium.beginFlight({Point3(4,0,4),Vector3::UnitZ()}),"domain miss does not start first-hit inference");
    // Full 512x512 / 128 spp camera regression: pixel (503,467), sample 51.
    // Subtracting its camera entry/exit distances used to extend the tiny
    // grazing chord past the stored SDF face and create an uncovered sliver.
    GPSSField grazingField = field;
    grazingField.activeDomain = {
        Point3(-0.5996229963955557,-0.5996229963955557,-0.15318835674339015),
        Point3(0.5952461862028875,0.5952461862028875,0.9672750525796922)};
    const FirstHitSEMedium grazingMedium(grazingField,model);
    const Ray grazingCamera{Point3(2.27,-1.09,1.42),
        Vector3(-0.8255455045894271,0.16794554584114818,-0.5387661027610285)};
    const auto grazing = grazingMedium.beginFlight(grazingCamera);
    context.require(grazing.has_value(),"a nonempty grazing camera chord is retained");
    if (grazing) {
        const double end = grazing->mean.maximumX()*grazing->ell;
        const Point3 endpoint = grazing->ray.origin+end*grazing->ray.direction;
        context.require(grazingField.activeDomain.contains(endpoint),
            "first-hit endpoint stays inside the field after camera-to-entry cancellation");
        context.require(end <= grazingField.activeDomain.intersect(grazing->ray).exit,
            "first-hit profile never extends past the actual clipped ray exit");
        context.near(end,0.00012676549253257263,1e-15,"grazing chord is preserved to floating-point accuracy");
        context.near(grazing->entryDistance,2.919857154581932,1e-14,"grazing fix preserves distance from camera to entry");
    }
    const FunctionSpaceRaySampler sampler({});
    Random refRng(731), directRng(731), gradRng(92);
    const auto ref = reference.sample(flight,refRng,gradRng);
    const auto direct = sampleFunctionSpaceFirstPassage(sampler,flight.mean,FunctionSpaceStart::Unconditioned,0,directRng);
    context.require(ref.hit == direct.hit && ref.initialSign == direct.initialSign && ref.q == direct.distance &&
        ref.speed == direct.speed,"render GPIS adapter preserves reference event samples exactly");

    int hits = 0, negative = 0;
    for (int i = 0; i < 1024; ++i) {
        Random e(firstHitStreamSeed(123,0,i,1)), g(firstHitStreamSeed(123,0,i,2));
        const auto hit = medium.sample(flight,e,g);
        if (!hit.hit) continue;
        ++hits; negative += hit.initialSign < 0;
        const double expected = -hit.initialSign*field.kernel.sigma()*hit.speed;
        context.near(flight.ray.direction.dot(hit.gradient),expected,1e-14,"full gradient keeps signed crossing speed");
    }
    const double probability = 1-std::exp(-std::log(2.0)*flight.mean.maximumX());
    context.near(hits/1024.0,probability,0.05,"constant RNO hazard has analytic hit probability");
    context.require(negative > hits*0.4 && negative < hits*0.6,"both entry signs survive sampling");
    auto plane = field; plane.mean = std::make_shared<PlaneMean>(Vector3::UnitZ(),0.2);
    const auto negativeField = firstHitSideField(plane,-1);
    const auto jet = plane.mean->evaluate(Point3(0,0,0.6));
    const auto njet = negativeField.mean->evaluate(Point3(0,0,0.6));
    context.near(njet.value,-jet.value,0,"negative-side mean is negated with its gradient");
    context.near((njet.gradient+jet.gradient).norm(),0,0,"negative-side full mean gradient");
    const auto negativeFlight = RayMeanProfile::pointLinear(negativeField.mean,Point3::Zero(),Vector3::UnitZ(),0.1,1,0.5,0.1);
    context.near(negativeFlight.segment(0).derivative(0),-10,1e-12,"side transform also preserves lazy point-linear jets");

    ExperimentConfig config;
    config.transportMode = "hybrid_first_hit"; config.field = field;
    config.firstHit.model = model; config.renewal.model = renewal;
    config.renewal.profileMode = "point_linear";
    config.material.ndfFamily = NdfFamily::GeneralizedGaussian;
    config.material.conductor.forceUnitFresnel = true;
    config.render.cameraPosition = camera.origin; config.render.cameraTarget = Point3::Zero();
    config.render.verticalFovDegrees = 20; config.render.width = 4; config.render.height = 4;
    config.render.samplesPerPixel = 8; config.render.safetyDepthCap = 128;
    config.render.rouletteStartDepth = 128; config.render.environment = "unit_white";
    config.render.writeFirstHitSamples = true; config.render.threadCount = 1;
    const auto serial = renderAnalyticScene(config);
    context.require(serial.statistics.firstHitUpcrossings > 0,"upcrossing paths enter signed renewal continuation");
    context.require(serial.statistics.safetyCapTerminations == 0 && serial.statistics.numericalFailures == 0,
        "hybrid fixture escapes without numerical failures or truncation");
    for (const auto& pixel : serial.pixels)
        context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"white furnace: no duplicate transmittance or hit-density weighting");
    config.render.threadCount = 3;
    const auto parallel = renderAnalyticScene(config);
    for (std::size_t i = 0; i < serial.pixels.size(); ++i)
        context.require(serial.pixels[i] == parallel.pixels[i],"hybrid render is independent of worker scheduling");
    config.render.safetyDepthCap = 1;
    const auto capped = renderAnalyticScene(config);
    context.require(capped.statistics.realCollisions == capped.statistics.firstHits &&
        capped.statistics.renewal.flights == 0,"depth cap includes the first collision and prevents secondary flights");
    for (std::size_t i = 0; i < serial.firstHitRecords.size(); ++i) {
        const auto& p = serial.firstHitRecords[i]; const auto& c = capped.firstHitRecords[i];
        context.require(p.cameraRay.direction == c.cameraRay.direction && p.first.hit == c.first.hit &&
            p.first.q == c.first.q && p.first.gradient == c.first.gradient,
            "camera and first-hit RNG do not depend on continuation work");
    }
    ExperimentConfig local = config;
    local.continuationMethod = "classic_local";
    local.renewal.model.reset();
    local.renewal.modelPath = "unused_missing_renewal_model.json";
    local.material.gpModel = GpModel::GlobalPointwise;
    local.preparedAreaMajorant = 1000.0;
    local.classicPhaseProposal = "target_vndf";
    local.mediumDensity = std::make_shared<LocalTestDensity>();
    local.densityMajorantGrid = std::make_shared<DensityMajorantGrid>(*local.mediumDensity,field.activeDomain,4);
    prepareRenewalModel(local);
    context.require(local.material.gpModel == GpModel::LocalTangent && !local.preparedAreaMajorant && !local.renewal.model,
        "local continuation selects its own material and bound without loading a renewal checkpoint");
    const auto cappedLocal = renderAnalyticScene(local);
    context.require(cappedLocal.statistics.classicFlights == 0 && cappedLocal.statistics.renewal.flights == 0 &&
        cappedLocal.statistics.realCollisions == cappedLocal.statistics.firstHits,
        "local continuation respects the first-collision depth cap");
    for (std::size_t i = 0; i < capped.pixels.size(); ++i)
        context.require(cappedLocal.pixels[i] == capped.pixels[i],"first reflection is independent of continuation method");
    local.render.safetyDepthCap = 128;
    local.render.threadCount = 1;
    const auto localSerial = renderAnalyticScene(local);
    context.require(localSerial.statistics.realCollisions > localSerial.statistics.firstHits &&
        localSerial.statistics.classicFlights > 0 && localSerial.statistics.tracking.candidates > 0 &&
        localSerial.statistics.renewal.flights == 0,"only local DDA samples flights after the neural first hit");
    context.require(localSerial.statistics.numericalFailures == 0 && localSerial.statistics.safetyCapTerminations == 0 &&
        localSerial.statistics.firstHitUpcrossings > 0,"local tail handles both first-hit signs without failure");
    for (const auto& pixel : localSerial.pixels)
        context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"neural/local white furnace has no extra hit weighting");
    local.render.threadCount = 3;
    const auto localParallel = renderAnalyticScene(local);
    for (std::size_t i = 0; i < localSerial.pixels.size(); ++i)
        context.require(localSerial.pixels[i] == localParallel.pixels[i],"neural/local render is independent of worker scheduling");
    for (std::size_t i = 0; i < serial.firstHitRecords.size(); ++i) {
        const auto& a = serial.firstHitRecords[i]; const auto& b = localSerial.firstHitRecords[i];
        context.require(a.cameraRay.direction == b.cameraRay.direction && a.first.hit == b.first.hit &&
            a.first.initialSign == b.first.initialSign && a.first.q == b.first.q && a.first.gradient == b.first.gradient,
            "switching to local continuation preserves camera and first-hit samples exactly");
    }
    // Check attachment against standalone classic_local paths, including first
    // Fresnel and a nonconstant environment. No roulette occurs at these depths.
    local.material.conductor.forceUnitFresnel = false;
    local.render.environment = "directional_gradient";
    const auto colored = renderAnalyticScene(local);
    ExperimentConfig tail = local;
    tail.transportMode = "classic_local";
    --tail.render.safetyDepthCap; --tail.render.rouletteStartDepth;
    std::uint64_t tailCollisions = 0;
    for (std::size_t i = 0; i < colored.firstHitRecords.size(); ++i) {
        const auto& r = colored.firstHitRecords[i];
        if (!r.first.hit) continue;
        const Ray reflected{r.first.position,normalizedOrThrow(reflectTravelDirection(r.cameraRay.direction,r.first.normal))};
        Random tailRng(firstHitStreamSeed(local.seed,i/local.render.samplesPerPixel,i%local.render.samplesPerPixel,3));
        RenderStatistics stats;
        const Spectrum expected = conductorFresnel(-r.cameraRay.direction.dot(r.first.normal),local.material.conductor)
            .cwiseProduct(traceCameraPath(reflected,tail,tailRng,stats));
        tailCollisions += stats.realCollisions;
        context.near((expected-r.radiance).norm(),0,1e-13,"local tail starts at the first reflection with its Fresnel weight");
        context.require(stats.rouletteTerminations == 0 && stats.safetyCapTerminations == 0,
            "standalone comparison needs no roulette or truncation correction");
    }
    context.require(colored.statistics.realCollisions == colored.statistics.firstHits+tailCollisions,
        "hybrid collision statistics count the first reflection exactly once");
    writeResolvedConfig(local,path);
    Json resolved; { std::ifstream f(path); f >> resolved; }
    std::filesystem::remove(path);
    context.require(resolved["transport"]["continuation"]["method"] == "classic_local" &&
        !resolved["transport"].contains("renewal") && !resolved["transport"]["continuation"].contains("model"),
        "resolved local config contains no unused renewal model");
    config.firstHit.method = "gpis_function_space";
    const auto gpis = renderAnalyticScene(config);
    for (std::size_t i = 0; i < serial.firstHitRecords.size(); ++i)
        context.require(gpis.firstHitRecords[i].cameraRay.direction == serial.firstHitRecords[i].cameraRay.direction,
            "RNO and GPIS render exactly the same camera rays");
    config.firstHit.backend = "auto";
    context.require(rejects([&] { prepareFirstHitModel(config); }),"unsupported first-hit backend is rejected");
    config.firstHit.backend = "scalar"; config.firstHit.profileMode = "point_linear";
    context.require(rejects([&] { prepareFirstHitModel(config); }),"first-hit training profile contract is enforced");
    config.continuationMethod = "classic_global";
    context.require(rejects([&] { prepareRenewalModel(config); }),"unimplemented continuation cannot silently fall back");
}

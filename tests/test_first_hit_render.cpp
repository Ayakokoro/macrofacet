#include "TestHarness.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
using Json = nlohmann::json;
class LocalTestDensity final : public mf::ScalarField {
    double value_;
public:
    explicit LocalTestDensity(double value = 8.0) : value_(value) {}
    double sample(const mf::Point3&) const override { return value_; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override { return {value_,value_,true}; }
};
class LocalFirstTestMean final : public mf::MeanField {
public:
    const char* typeName() const override { return "local_first_test"; }
    mf::MeanJet evaluate(const mf::Point3& p) const override { return {.25*p.z(),{0,0,.25}}; }
    mf::BoundsSummary bounds(const mf::Bounds3& b) const override { return {.25*b.minimum.z(),.25,true}; }
    std::optional<mf::Vector3> affineGradient() const override { return mf::Vector3(0,0,.25); }
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

void testLocalFirstHit(TestContext& context) {
    using namespace mf;
    ExperimentConfig config;
    config.transportMode = "hybrid_first_hit";
    config.firstHit.method = "classic_local";
    config.continuationMethod = kTungstenFunctionSpaceMethod;
    config.continuationTungsten = {8,.15};
    // The global mean gradient is shorter than one, so its valid area bound
    // cannot be reused for the local tangent NDF at the first collision.
    config.field = {std::make_shared<LocalFirstTestMean>(),
        CovarianceKernel::fromCorrelationLengths(.1,Vector3::Ones()),
        {Point3::Constant(-1),Point3::Constant(1)}};
    config.material.gpModel = GpModel::GlobalPointwise;
    config.mediumDensity = std::make_shared<LocalTestDensity>(.4);
    config.densityMajorantGrid = std::make_shared<DensityMajorantGrid>(*config.mediumDensity,config.field.activeDomain,4);
    config.preparedAreaMajorant = classicAreaMajorant(config.field,config.material,config.field.activeDomain,Vector3::UnitZ());
    config.render.cameraPosition = Point3(0,0,4);
    config.render.cameraTarget = Point3::Zero();
    config.render.verticalFovDegrees = 20;
    config.render.width = 4; config.render.height = 4; config.render.samplesPerPixel = 16;
    config.render.threadCount = 1; config.render.safetyDepthCap = 2;
    config.render.writeFirstHitSamples = true;
    config.render.environment = "directional_gradient";
    const auto environment = [](const Vector3& w) -> Spectrum {
        const double up = .5*(normalizedOrThrow(w).z()+1);
        return {0.15+0.85*up,0.2+0.7*up,0.35+0.55*up};
    };
    auto localMaterial = config.material;
    localMaterial.gpModel = GpModel::LocalTangent;
    const NarrowBandMedium first(config.field,localMaterial,config.mediumDensity,config.densityMajorantGrid);
    const FunctionSpaceGpisMedium tail(config.field,config.continuationTungsten);
    for (int rouletteStart : {100,1}) {
        config.render.rouletteStartDepth = rouletteStart;
        const auto image = renderAnalyticScene(config);
        std::uint64_t firstHits = 0, tailFlights = 0, tailEscapes = 0;
        for (std::size_t i=0; i<image.firstHitRecords.size(); ++i) {
            const auto& record = image.firstHitRecords[i];
            const auto& actual = record.first;
            const auto pixel = i/config.render.samplesPerPixel, sample = i%config.render.samplesPerPixel;
            Random event(firstHitStreamSeed(config.seed,pixel,sample,1));
            Random gradient(firstHitStreamSeed(config.seed,pixel,sample,2));
            Random continuation(firstHitStreamSeed(config.seed,pixel,sample,3));
            const auto seed = continuation.integer();
            context.require(seed==record.gpisSeed,"local first-hit leaves the GPIS random stream independent");
            const auto& ray = record.cameraRay;
            const auto interval = config.field.activeDomain.intersect(ray);
            const Point3 entry = (ray.origin+interval.entry*ray.direction)
                .cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
            const auto kernel = first.beginFlight(first.startExternal(entry,ray.direction));
            const auto expectedHit = first.sample(kernel,event,config.numeric);
            context.require(actual.enteredDomain && actual.hit==expectedHit.collided && actual.initialSign==1,
                "local first-hit reproduces the standalone local DDA collision law");
            context.near(actual.entryDistance,interval.entry,0,"local first-hit separates the camera vacuum distance");
            context.near(actual.distance,expectedHit.age,0,"local first-hit distance starts at the medium entry");
            context.near((actual.position-(entry+expectedHit.age*ray.direction)).norm(),0,0,
                "local first-hit reconstructs world positions from entry age");
            if (!actual.hit) {
                context.near((record.radiance-environment(ray.direction)).norm(),0,1e-14,
                    "local first-hit escape returns the unweighted environment");
                continue;
            }
            ++firstHits;
            const Vector3 g = sampleCollisionGradient(kernel,expectedHit.age,gradient,config.numeric);
            context.near((actual.gradient-g).norm(),0,0,
                "local first-hit retains the full local flux-weighted gradient rather than a unit normal");
            context.require(actual.fieldValue==0 && actual.observationPosition==actual.position &&
                ray.direction.dot(g)<0,"local first-hit exports a downcrossing surface state");
            Random phaseRng(firstHitStreamSeed(config.seed,pixel,sample,2));
            const auto phase = ConductorPhase(config.field,localMaterial,actual.position,0,true).samplePhase(ray.direction,phaseRng);
            const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(ray.direction,actual.normal));
            context.near((outgoing-phase.direction).norm(),0,1e-13,
                "local first-hit reflection matches the standalone local target VNDF");
            context.require(outgoing.dot(g)>0,"local reflection leaves the positive side");
            Spectrum weight = phase.throughputWeight;
            TungstenRandom rng(seed);
            if (rouletteStart==1) {
                const double probability = std::clamp(weight.maxCoeff(),.05,.95);
                if (rng.uniform01()>=probability) {
                    context.near(record.radiance.norm(),0,0,"local first-hit roulette precedes GPIS continuation");
                    continue;
                }
                weight /= probability;
            }
            ++tailFlights;
            const auto draw = tail.sample({actual.position,outgoing},{TungstenSurfaceState{actual.position,0,g}},rng);
            Spectrum expected = Spectrum::Zero();
            if (draw.status==GpisFlightStatus::Escape) {
                ++tailEscapes;
                expected = weight.cwiseProduct(environment(outgoing));
            }
            context.near((record.radiance-expected).norm(),0,1e-13,
                "local-to-GPIS handoff matches surface conditioning and a single local phase weight");
        }
        context.require(firstHits>0 && firstHits<image.statistics.paths && tailEscapes>0,
            "local first-hit fixture covers collisions, camera escapes and GPIS escapes");
        context.require(image.statistics.classicFirstHitFlights==image.statistics.firstHitFlights &&
            image.statistics.classicFlights==image.statistics.classicFirstHitFlights &&
            image.statistics.gpisContinuationFlights==tailFlights && image.statistics.gpisFlights==tailFlights &&
            image.statistics.conditionalFlights==0,"local first-hit and GPIS continuation counters remain separate");
        context.require(image.statistics.firstHitUpcrossings==0 && image.statistics.numericalFailures==0,
            "local first-hit uses valid local bounds and only facing gradients");
        config.render.threadCount = 3;
        const auto parallel = renderAnalyticScene(config);
        context.require(image.pixels==parallel.pixels &&
            image.statistics.classicFirstHitFlights==parallel.statistics.classicFirstHitFlights,
            "local first-hit pixels and diagnostics are independent of worker scheduling");
        config.render.threadCount = 1;
        auto otherTail = config;
        otherTail.continuationMethod = "classic_local";
        const auto other = renderAnalyticScene(otherTail);
        context.require(other.statistics.classicFirstHitFlights==image.statistics.firstHitFlights &&
            other.statistics.gpisFlights==0,"local/local diagnostics retain the first-flight count without GPIS work");
        if (rouletteStart==100)
            context.require(other.statistics.classicFlights==other.statistics.classicFirstHitFlights+firstHits,
                "local/local diagnostics count each first and continuation flight");
        for (std::size_t i=0; i<image.firstHitRecords.size(); ++i) {
            const auto& a = image.firstHitRecords[i].first;
            const auto& b = other.firstHitRecords[i].first;
            context.require(a.hit==b.hit && a.distance==b.distance && a.gradient==b.gradient,
                "local first-hit material and random samples do not depend on the continuation strategy");
        }
    }
    Random missRng(19); RenderStatistics missStats;
    const auto miss = traceCameraPath({Point3(3,0,4),Vector3::UnitZ()},config,missRng,missStats);
    context.require(missStats.firstHitDomainMisses==1 && missStats.classicFlights==0 && missStats.gpisFlights==0,
        "domain misses skip both local tracking and GPIS continuation");
    context.near((miss-environment(Vector3::UnitZ())).norm(),0,0,"local first-hit domain miss returns environment");
    config.firstHit.backend = "torch_cuda";
    context.require(rejects([&] { prepareFirstHitModel(config); }),"local first-hit rejects non-scalar backends");
    config.firstHit.backend = "scalar";
    config.material.ndfFamily = NdfFamily::GGXBaseline;
    context.require(rejects([&] { prepareFirstHitModel(config); }),"local first-hit needs a Gaussian full gradient");
    config.material.ndfFamily = NdfFamily::GeneralizedGaussian;
    config.material.alphaField = std::make_shared<LocalTestDensity>();
    context.require(rejects([&] { prepareFirstHitModel(config); }),"local first-hit rejects alpha grids");
}

void testConditionalFirstHit(TestContext& context, double renewalDistance = 0.0) {
    using namespace mf;
    ExperimentConfig config;
    config.transportMode = "hybrid_first_hit";
    const bool renew = renewalDistance > 0.0;
    config.firstHit.method = renew ? "global_conditional_renewal_plus" : "global_conditional";
    if (renew) config.firstHit.renewalDistance = renewalDistance;
    config.continuationMethod = kTungstenFunctionSpaceMethod;
    config.continuationTungsten = {8,.15};
    // A negative mean must not be treated as an invalid exterior: F is random.
    config.field = {std::make_shared<ConstantMean>(-.03),
        CovarianceKernel::fromCorrelationLengths(.1,Vector3::Ones()),
        {Point3::Constant(-1),Point3::Constant(1)}};
    config.material.ndfFamily = NdfFamily::GeneralizedGaussian;
    config.render.cameraPosition = Point3(0,0,4);
    config.render.cameraTarget = Point3::Zero();
    config.render.verticalFovDegrees = 20;
    config.render.width = 4; config.render.height = 4; config.render.samplesPerPixel = 16;
    config.render.threadCount = 1; config.render.safetyDepthCap = 2;
    config.render.writeFirstHitSamples = true;
    config.render.environment = "directional_gradient";
    const auto environment = [](const Vector3& w) -> Spectrum {
        const double up = .5*(normalizedOrThrow(w).z()+1);
        return {0.15+0.85*up,0.2+0.7*up,0.35+0.55*up};
    };
    const ConditionalMedium first(config.field);
    const GlobalConditionalRenewalPlusMedium renewedFirst(config.field,renew ? renewalDistance : .1);
    const FunctionSpaceGpisMedium tail(config.field,config.continuationTungsten);
    for (int rouletteStart : {100,1}) {
        config.render.rouletteStartDepth = rouletteStart;
        const auto image = renderAnalyticScene(config);
        std::uint64_t firstHits = 0, tailFlights = 0, tailEscapes = 0;
        std::uint64_t expectedRenewalSegments = 0;
        for (std::size_t i=0; i<image.firstHitRecords.size(); ++i) {
            const auto& record = image.firstHitRecords[i];
            const auto& actual = record.first;
            const auto pixel = i/config.render.samplesPerPixel, sample = i%config.render.samplesPerPixel;
            Random event(firstHitStreamSeed(config.seed,pixel,sample,1));
            Random gradient(firstHitStreamSeed(config.seed,pixel,sample,2));
            Random continuation(firstHitStreamSeed(config.seed,pixel,sample,3));
            const auto seed = continuation.integer();
            context.require(seed==record.gpisSeed,"conditional first-hit keeps the GPIS stream independent");
            const auto& ray = record.cameraRay;
            const auto interval = config.field.activeDomain.intersect(ray);
            const Point3 entry = (ray.origin+interval.entry*ray.direction)
                .cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
            const auto birth = first.startExternal(entry,ray.direction,event);
            context.require(birth.birthValue>0 && birth.birthGradient.allFinite(),
                "conditional camera samples a positive total field even with a negative mean SDF");
            const auto kernel = first.beginFlight(birth);
            GlobalConditionalRenewalPlusSample expectedHit;
            if (renew && renewalDistance < kernel.maximumAgeInDomain()) {
                expectedHit = renewedFirst.sample(birth,event,nullptr,-1,&expectedRenewalSegments,
                    config.numeric,&gradient);
            } else {
                // A single renewal interval must exactly reproduce the original
                // conditional first hit, including its separate gradient stream.
                const auto draw = first.sample(kernel,event);
                expectedHit.collided = draw.collided;
                expectedHit.age = draw.age;
                if (draw.collided)
                    expectedHit.gradient = sampleCollisionGradient(kernel,draw.age,gradient,config.numeric);
                if (renew) ++expectedRenewalSegments;
            }
            context.require(actual.enteredDomain && actual.initialSign==1 && actual.hit==expectedHit.collided,
                "conditional first-hit uses the ordinary positive-exterior collision law");
            context.near(actual.entryDistance,interval.entry,0,"conditional first-hit separates the camera vacuum distance");
            context.near(actual.distance,expectedHit.age,0,"conditional first-hit age starts at the medium entry");
            context.near((actual.position-(entry+expectedHit.age*ray.direction)).norm(),0,0,
                "conditional first-hit reconstructs world positions from entry age");
            if (!actual.hit) {
                context.near((record.radiance-environment(ray.direction)).norm(),0,1e-14,
                    "positive-exterior escape is not multiplied by the prior probability of F>0");
                continue;
            }
            ++firstHits;
            const Vector3 g = expectedHit.gradient;
            context.near((actual.gradient-g).norm(),0,0,
                "first-hit gradient retains the conditional crossing flux and full magnitude");
            context.require(actual.fieldValue==0 && actual.observationPosition==actual.position &&
                ray.direction.dot(actual.gradient)<0,"conditional first-hit exports a downcrossing surface state");
            const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(ray.direction,normalizedOrThrow(g)));
            context.require(outgoing.dot(g)>0,"conditional reflection leaves the positive side");
            Spectrum weight = conductorFresnel(-ray.direction.dot(normalizedOrThrow(g)),config.material.conductor);
            TungstenRandom rng(seed);
            if (rouletteStart==1) {
                const double probability = std::clamp(weight.maxCoeff(),.05,.95);
                if (rng.uniform01()>=probability) {
                    context.near(record.radiance.norm(),0,0,"first-hit roulette terminates before GPIS continuation");
                    continue;
                }
                weight /= probability;
            }
            ++tailFlights;
            const auto draw = tail.sample({actual.position,outgoing},{TungstenSurfaceState{actual.position,0,g}},rng);
            Spectrum expected = Spectrum::Zero();
            if (draw.status==GpisFlightStatus::Escape) {
                ++tailEscapes;
                expected = weight.cwiseProduct(environment(outgoing));
            }
            context.near((record.radiance-expected).norm(),0,1e-13,
                "conditional-to-GPIS handoff matches surface conditioning with one Fresnel factor");
        }
        context.require(firstHits>0 && firstHits<image.statistics.paths && tailEscapes>0,
            "conditional first-hit fixture exercises both hits and escapes");
        context.require(image.statistics.conditionalFirstHitFlights==image.statistics.firstHitFlights &&
            image.statistics.conditionalFlights==image.statistics.conditionalFirstHitFlights &&
            image.statistics.gpisContinuationFlights==tailFlights && image.statistics.gpisFlights==tailFlights,
            "conditional first-hit and GPIS continuation counts remain separate");
        context.require(image.statistics.firstHitUpcrossings==0 && image.statistics.numericalFailures==0,
            "conditional first-hit is positive-side only and has no numerical failures");
        context.require(image.statistics.conditionalFirstHitRenewalSegments==expectedRenewalSegments &&
            image.statistics.conditionalRenewalSegments==expectedRenewalSegments,
            "first-hit renewal counters match standalone segments and exclude GPIS flights");
        if (renew && renewalDistance==.1)
            context.require(expectedRenewalSegments>image.statistics.conditionalFirstHitFlights,
                "renewal first-hit fixture advances through multiple positive endpoints");
        config.render.threadCount = 3;
        const auto parallel = renderAnalyticScene(config);
        context.require(image.pixels==parallel.pixels &&
            image.statistics.conditionalFirstHitFlights==parallel.statistics.conditionalFirstHitFlights &&
            image.statistics.conditionalFirstHitRenewalSegments==parallel.statistics.conditionalFirstHitRenewalSegments,
            "conditional first-hit pixels and diagnostics are independent of worker scheduling");
        config.render.threadCount = 1;
        auto otherTail = config;
        otherTail.continuationMethod = "global_conditional";
        otherTail.conditionalRenewalDistance = .037;
        const auto other = renderAnalyticScene(otherTail);
        for (std::size_t i=0; i<image.firstHitRecords.size(); ++i) {
            const auto& a = image.firstHitRecords[i].first;
            const auto& b = other.firstHitRecords[i].first;
            context.require(a.hit==b.hit && a.distance==b.distance && a.gradient==b.gradient,
                "changing continuation does not alter conditional first-hit samples");
        }
    }
    Random missRng(19); RenderStatistics missStats;
    const auto miss = traceCameraPath({Point3(3,0,4),Vector3::UnitZ()},config,missRng,missStats);
    context.require(missStats.firstHitDomainMisses==1 && missStats.conditionalFlights==0 && missStats.gpisFlights==0,
        "domain misses do not sample conditional births or GPIS flights");
    context.near((miss-environment(Vector3::UnitZ())).norm(),0,0,"domain miss returns environment");
    config.firstHit.backend = "torch_cuda";
    context.require(rejects([&] { prepareFirstHitModel(config); }),"conditional first-hit rejects non-scalar inference backends");
    config.firstHit.backend = "scalar";
    config.material.alphaField = std::make_shared<LocalTestDensity>();
    context.require(rejects([&] { prepareFirstHitModel(config); }),"conditional first-hit rejects alpha grids");
    config.material.alphaField.reset();
    if (renew) {
        for (double invalid : {0.0,-.1,std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            auto bad = config;
            bad.firstHit.renewalDistance = invalid;
            context.require(rejects([&] { prepareFirstHitModel(bad); }),
                "conditional first-hit renewal distance must be finite and positive");
        }
    }
    config.field.kernel = CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::Matern32,.1,Vector3::Ones());
    context.require(rejects([&] { prepareFirstHitModel(config); }),"conditional first-hit rejects unsupported kernels");
}
}

void testFirstHitRender(TestContext& context) {
    using namespace mf;
    testLocalFirstHit(context);
    testConditionalFirstHit(context);
    testConditionalFirstHit(context,.1);
    testConditionalFirstHit(context,10.0);
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
    context.near((ref.observationPosition-ref.position).norm(),0,0,
        "legacy first-hit exposes its observation position for independent continuation");
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
    auto tungstenTail=config;
    tungstenTail.continuationMethod=kTungstenFunctionSpaceMethod;
    tungstenTail.continuationTungsten={8,.15};
    tungstenTail.render.safetyDepthCap=12;
    const auto tungstenMixed=renderAnalyticScene(tungstenTail);
    context.require(tungstenMixed.statistics.gpisContinuationFlights>0,
        "neural first-hit feeds the Tungsten continuation without loading a renewal model");
    for (std::size_t i=0;i<serial.firstHitRecords.size();++i) {
        context.near((serial.firstHitRecords[i].first.position-tungstenMixed.firstHitRecords[i].first.position).norm(),0,0,
            "Tungsten continuation preserves neural first-hit positions");
        context.near((serial.firstHitRecords[i].first.gradient-tungstenMixed.firstHitRecords[i].first.gradient).norm(),0,0,
            "Tungsten continuation preserves neural first-hit gradient random stream");
    }
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

    ExperimentConfig conditionalRenewal = config;
    conditionalRenewal.continuationMethod="global_conditional_renewal_plus";
    conditionalRenewal.conditionalRenewalDistance=0.05;
    conditionalRenewal.renewal.model.reset();
    conditionalRenewal.renewal.modelPath="unused_missing_renewal_model.json";
    conditionalRenewal.render.safetyDepthCap=128;
    conditionalRenewal.render.rouletteStartDepth=1000;
    conditionalRenewal.render.threadCount=1;
    prepareRenewalModel(conditionalRenewal);
    const auto conditionalRenewalImage=renderAnalyticScene(conditionalRenewal);
    context.require(conditionalRenewalImage.statistics.conditionalFlights>0 &&
        conditionalRenewalImage.statistics.conditionalRenewalSegments>0 &&
        conditionalRenewalImage.statistics.renewal.flights==0 &&
        conditionalRenewalImage.statistics.numericalFailures==0 &&
        conditionalRenewalImage.statistics.safetyCapTerminations==0,
        "hybrid global conditional renewal+ continues from the first surface without neural checkpoints");
    for (std::size_t i=0;i<serial.firstHitRecords.size();++i) {
        const auto& a=serial.firstHitRecords[i]; const auto& b=conditionalRenewalImage.firstHitRecords[i];
        context.require(a.first.hit==b.first.hit && a.first.initialSign==b.first.initialSign &&
            a.first.gradient==b.first.gradient && a.first.q==b.first.q,
            "global conditional renewal+ handoff preserves both signs and the original first-hit observations");
    }
    for (const auto& pixel : conditionalRenewalImage.pixels)
        context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"hybrid global conditional renewal+ white furnace has no extra phase weight");

    ExperimentConfig conditional = config;
    conditional.continuationMethod = "global_conditional";
    conditional.renewal.model.reset();
    conditional.renewal.modelPath = "unused_missing_renewal_model.json";
    conditional.material.gpModel = GpModel::LocalTangent;
    conditional.preparedAreaMajorant = 1000.0;
    prepareRenewalModel(conditional);
    context.require(conditional.material.gpModel == GpModel::GlobalPointwise &&
        !conditional.preparedAreaMajorant && !conditional.renewal.model,
        "conditional continuation selects the global GP without loading a renewal model");
    const auto cappedConditional = renderAnalyticScene(conditional);
    context.require(cappedConditional.statistics.conditionalFlights == 0 &&
        cappedConditional.statistics.realCollisions == cappedConditional.statistics.firstHits,
        "conditional continuation respects the first-collision depth cap");
    for (std::size_t i = 0; i < capped.pixels.size(); ++i)
        context.require(cappedConditional.pixels[i] == capped.pixels[i],
            "conditional continuation does not alter the first reflection");
    conditional.render.safetyDepthCap = 128;
    conditional.render.threadCount = 1;
    const auto conditionalSerial = renderAnalyticScene(conditional);
    context.require(conditionalSerial.statistics.conditionalFlights > 0 &&
        conditionalSerial.statistics.realCollisions > conditionalSerial.statistics.firstHits &&
        conditionalSerial.statistics.tracking.candidates > 0 &&
        conditionalSerial.statistics.classicFlights == 0 && conditionalSerial.statistics.renewal.flights == 0,
        "only conditional tracking samples the hybrid continuation");
    context.require(conditionalSerial.statistics.firstHitUpcrossings > 0 &&
        conditionalSerial.statistics.numericalFailures == 0 && conditionalSerial.statistics.safetyCapTerminations == 0,
        "conditional continuation handles both first-hit signs without failure");
    for (const auto& pixel : conditionalSerial.pixels)
        context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,
            "neural/conditional white furnace preserves energy through multiple reflections");
    for (std::size_t i = 0; i < serial.firstHitRecords.size(); ++i) {
        const auto& a = serial.firstHitRecords[i]; const auto& b = conditionalSerial.firstHitRecords[i];
        context.require(a.cameraRay.direction == b.cameraRay.direction && a.first.hit == b.first.hit &&
            a.first.initialSign == b.first.initialSign && a.first.q == b.first.q && a.first.gradient == b.first.gradient,
            "conditional continuation preserves the camera and first-hit random streams");
    }
    conditional.render.threadCount = 3;
    const auto conditionalParallel = renderAnalyticScene(conditional);
    for (std::size_t i = 0; i < conditionalSerial.pixels.size(); ++i)
        context.require(conditionalSerial.pixels[i] == conditionalParallel.pixels[i],
            "hybrid conditional pixels are independent of worker scheduling");
    context.require(conditionalSerial.statistics.conditionalFlights == conditionalParallel.statistics.conditionalFlights &&
        conditionalSerial.statistics.realCollisions == conditionalParallel.statistics.realCollisions &&
        conditionalSerial.statistics.tracking.candidates == conditionalParallel.statistics.tracking.candidates,
        "hybrid conditional statistics merge across workers");

    conditional.firstHit.method = "gpis_function_space";
    // A nonzero mean makes negating the field (not just the gradient) observable.
    conditional.field.mean = std::make_shared<ConstantMean>(0.03);
    const auto gpisConditional = renderAnalyticScene(conditional);
    context.require(gpisConditional.statistics.firstHitUpcrossings > 0 &&
        gpisConditional.statistics.firstHitUpcrossings < gpisConditional.statistics.firstHits &&
        gpisConditional.statistics.conditionalFlights > 0 &&
        gpisConditional.statistics.realCollisions > gpisConditional.statistics.firstHits &&
        gpisConditional.statistics.safetyCapTerminations == 0 && gpisConditional.statistics.numericalFailures == 0,
        "GPIS first hits of both signs feed multiple conditional flights");
    for (const auto& pixel : gpisConditional.pixels)
        context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,"GPIS/conditional white furnace preserves energy");

    // With a two-reflection cap, only escape on the first continuation flight
    // contributes. Compare each path to the standalone conditional sampler;
    // this detects exterior reinitialization, lost gradient magnitude, wrong
    // side/mean, duplicate Fresnel and incorrect first-collision roulette.
    conditional.material.conductor.forceUnitFresnel = false;
    conditional.render.environment = "directional_gradient";
    conditional.render.safetyDepthCap = 2;
    for (int rouletteStart : {128,1}) {
        conditional.render.rouletteStartDepth = rouletteStart;
        const auto twoReflections = renderAnalyticScene(conditional);
        std::uint64_t expectedFlights = 0, tailHits = 0, tailEscapes = 0;
        for (std::size_t i = 0; i < twoReflections.firstHitRecords.size(); ++i) {
            const auto& r = twoReflections.firstHitRecords[i];
            if (!r.first.hit) continue;
            const auto& original = gpisConditional.firstHitRecords[i].first;
            context.require(r.first.q == original.q && r.first.gradient == original.gradient,
                "GPIS first-hit samples are independent of continuation depth and roulette");
            Random rng(firstHitStreamSeed(conditional.seed,i/conditional.render.samplesPerPixel,
                i%conditional.render.samplesPerPixel,3));
            Spectrum weight = conductorFresnel(-r.cameraRay.direction.dot(r.first.normal),conditional.material.conductor);
            if (rouletteStart == 1) {
                const double probability = std::clamp(weight.maxCoeff(),0.05,0.95);
                if (rng.openUniform01() >= probability) {
                    context.near(r.radiance.norm(),0,0,"first-collision roulette terminates before conditional sampling");
                    continue;
                }
                weight /= probability;
            }
            ++expectedFlights;
            GPSSField signedField = conditional.field;
            signedField.mean = std::make_shared<ConstantMean>(r.first.initialSign*0.03);
            const ConditionalMedium tailMedium(signedField);
            FlightState birth;
            birth.birthPosition = r.first.position;
            const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(r.cameraRay.direction,r.first.normal));
            birth.direction = normalizedOrThrow(outgoing);
            birth.birthGradient = r.first.initialSign*r.first.gradient;
            const auto event = tailMedium.sample(tailMedium.beginFlight(birth),rng);
            Spectrum expected = Spectrum::Zero();
            if (!event.collided) {
                ++tailEscapes;
                const double up = 0.5*(normalizedOrThrow(birth.direction).z()+1.0);
                expected = weight.cwiseProduct(Spectrum(0.15+0.85*up,0.2+0.7*up,0.35+0.55*up));
            } else ++tailHits;
            context.near((expected-r.radiance).norm(),0,1e-13,
                "GPIS conditional handoff matches the signed surface-conditioned escape sample");
        }
        context.require(expectedFlights == twoReflections.statistics.conditionalFlights && tailHits > 0 && tailEscapes > 0,
            "first-hit roulette and depth count exactly the conditional flights sampled");
    }
    writeResolvedConfig(conditional,path);
    { std::ifstream f(path); f >> resolved; }
    std::filesystem::remove(path);
    context.require(resolved["transport"]["continuation"]["method"] == "global_conditional" &&
        resolved["transport"]["continuation"]["birth_policy"] == "first_hit_surface" &&
        resolved["transport"]["continuation"]["sampler"] == "segmented_delta_tracking" &&
        !resolved["transport"].contains("renewal") && !resolved["transport"]["continuation"].contains("model"),
        "resolved conditional continuation records surface conditioning without a renewal checkpoint");
    conditional.renewal.backend = "auto";
    context.require(rejects([&] { prepareRenewalModel(conditional); }),"hybrid conditional backend cannot silently fall back");
    conditional.renewal.backend = "scalar";
    conditional.material.alphaField = std::make_shared<LocalTestDensity>();
    context.require(rejects([&] { prepareRenewalModel(conditional); }),"conditional continuation rejects an alpha grid");
    conditional.material.alphaField.reset();
    conditional.field.kernel = CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::Matern32,0.1,Vector3::Ones());
    context.require(rejects([&] { prepareRenewalModel(conditional); }),"conditional continuation rejects unsupported kernels");

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

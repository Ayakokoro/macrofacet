#include "TestHarness.h"
#include "macrofacet/transport/RenewalRayDistribution.h"
#include "macrofacet/transport/RenewalMedium.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
using Json = nlohmann::json;
Json constantBundle(bool full = false) {
    Json weights = Json::object();
    const auto tensor = [&](const std::string& name, std::vector<int> shape) {
        int size = 1;
        for (int s : shape) size *= s;
        weights[name] = {{"shape", shape}, {"values", std::vector<float>(size, 0.0f)}};
    };
    const auto linear = [&](const std::string& name, int input, int output) {
        tensor(name+".weight", {output, input}); tensor(name+".bias", {output});
    };
    linear("segment_encoder.0", 5, 3); linear("segment_encoder.2", 3, 3);
    linear("initial_encoder.0", 4, 2); linear("initial_encoder.2", 2, 2); linear("initial_encoder.4", 2, 2);
    linear("hazard_head.0", 5, 2); linear("hazard_head.2", 2, 2); linear("hazard_head.4", 2, 4);
    tensor("gru.weight_ih_l0", {6, 3}); tensor("gru.weight_hh_l0", {6, 2});
    tensor("gru.bias_ih_l0", {6}); tensor("gru.bias_hh_l0", {6});
    if (full) {
        linear("mixture_head.0", 8, 4); linear("mixture_head.2", 4, 4); linear("mixture_head.4", 4, 6);
    }
    Json result = {{"format", full ? "macrofacet.renewal" : "macrofacet.renewal_hazard"}, {"version", 1},
        {"kernel", {{"type", "matern_3_2"}, {"parameterization", "unit_decay"}, {"beta", 1.0}}},
        {"activation_dtype", "float32"}, {"gru_convention", "pytorch_rzn_reset_after"},
        {"feature_transform", "asinh_first_four_log_dx_identity"},
        {"initial_transform", "asinh_mode_b0_known_z0_known_d0"},
        {"checkpoint_sha256", "synthetic_constant_fixture"},
        {"model_config", {{"hidden", 2}, {"embedding", 3}, {"hazard_width", 2}}}, {"weights", weights}};
    if (full) {
        result["mixture_convention"] = "positive_truncated_components_residual_mean";
        result["model_config"]["mixture_width"] = 4;
        result["model_config"]["components"] = 2;
        result["model_config"]["sigma_floor"] = 0.01;
    }
    return result;
}
template<class F> bool rejects(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
}

void testRenewalTransport(TestContext& context) {
    using namespace mf;
    const RenewalHazardSegment segment(0, 0.3, {0.1, 3.0, 0.4, 2.0});
    context.near(segment.cumulative(0), 0, 0, "Renewal cumulative starts at zero");
    context.near(segment.cumulative(1), 0.3*(0.1+3+0.4+2)/4, 1e-15, "Bernstein full mass");
    double previous = 0;
    for (int i = 0; i <= 100; ++i) {
        const double u = i/100.0, value = segment.cumulative(u);
        context.require(value >= previous, "cumulative hazard is monotone");
        context.near(segment.integral(0, u), value, 5e-16, "positive interval integral equals cumulative polynomial");
        previous = value;
    }
    for (double u : {0.01, 0.1, 0.37, 0.8, 0.99}) {
        const double derivative = (segment.cumulative(u+1e-6)-segment.cumulative(u-1e-6))/(0.3*2e-6);
        context.near(derivative, segment.hazard(u), 1e-8, "hazard is the derivative in normalized distance");
        const double inverse = segment.inverse(segment.integral(0.005, u), 0.005, 1, 1e-12);
        context.near(inverse, u, 4e-12, "partial-segment cumulative inversion");
    }
    context.require(rejects([&] { segment.inverse(10, 0, 1, 1e-9); }), "reject optical depth beyond segment");
    context.require(rejects([] { RenewalHazardSegment(0, 1, {-1, 0, 0, 0}); }), "reject negative hazard coefficients");

    constexpr double ell = 0.2;
    const RenewalRayDistribution distribution({{0, 1, {0,0,0,0}}, {1,2,{2,2,2,2}},
                                                {2,3,{0,0,0,0}}, {3,4,{0.5,0.5,0.5,0.5}}}, ell);
    context.near(distribution.cumulativeHazard(4*ell), 2.5, 1e-14, "optical depth uses x=s/ell");
    context.near(distribution.hazard(1.5*ell), 10, 1e-14, "physical hazard divides by ell");
    const auto crossing = distribution.sampleOpticalDepth(1.4, 0, 4*ell, 1e-12);
    context.require(crossing.hit && crossing.segment == 1, "skip empty prefix and hit correct segment");
    context.near(crossing.distance, 1.7*ell, 1e-12, "sampled distance is physical");
    const auto miss = distribution.sampleOpticalDepth(3, 0, 3.5*ell);
    context.require(!miss.hit && miss.distance == 3.5*ell, "finite-cutoff miss retains remaining probability mass");
    context.near(distribution.transmittance(1.25*ell, 3.5*ell), std::exp(-1.75), 1e-14,
                 "survival-conditioned transmittance keeps birth history");
    context.near(distribution.transmittance(0.4, 0.4), 1, 0, "zero length interval is transparent");
    context.require(!distribution.sampleOpticalDepth(0, 0.4, 0.4).hit, "zero length interval cannot collide");
    context.require(rejects([&] { distribution.transmittance(-0.1); }), "reject invalid query range");
    context.require(rejects([&] { distribution.sampleOpticalDepth(-1, 0, 0.5); }), "reject negative exponential threshold");

    const RenewalRayDistribution opaque({{0,1,{1e20,1e20,1e20,1e20}}, {1,2,{1,1,1,1}}}, 1);
    context.near(opaque.transmittance(2), 0, 0, "unconditional transmittance may underflow to zero");
    context.near(opaque.transmittance(1, 1.5), std::exp(-0.5), 1e-15,
                 "conditional transmittance avoids cancellation of a huge prefix");
    const RenewalRayDistribution empty({{0,2,{0,0,0,0}}}, 1);
    context.near(empty.transmittance(2), 1, 0, "zero model hazard stays transparent without a floor");
    context.require(!empty.sampleOpticalDepth(0, 0, 2).hit, "zero threshold does not invent a hit in empty space");
    const RenewalHazardSegment huge(0, 1, {1e308,1e308,1e308,1e308});
    context.relative(huge.integral(0.2, 0.3), 1e307, 1e-14, "interval integration does not overflow intermediate sums");

    Random rng(8391);
    constexpr int trials = 32768;
    std::array<int, 4> survived{};
    for (int i = 0; i < trials; ++i) {
        const auto draw = distribution.sample(rng);
        for (int j = 0; j < 4; ++j)
            survived[j] += !draw.hit || draw.distance > (j+1)*ell;
    }
    for (int j = 0; j < 4; ++j) {
        const double p = distribution.transmittance((j+1)*ell);
        context.near(survived[j]/static_cast<double>(trials), p,
                     6*std::sqrt(p*(1-p)/trials)+0.001, "sample frequencies agree with model transmittance");
    }

    const auto path = std::filesystem::temp_directory_path()/"macrofacet_renewal_transport_fixture.json";
    auto bundle = constantBundle();
    { std::ofstream out(path); out << bundle; }
    const auto model = RenewalHazardModel::load(path);
    const auto mean = RayMeanProfile::affine(0.3, -0.4, 4, 0.25);
    for (const RayStartCondition start : {RayStartCondition{}, RayStartCondition{RayStartMode::SurfaceOutward, 0.7}}) {
        const auto cached = RenewalRayDistribution::fromModel(model, mean, start, ell);
        const double rate = static_cast<double>(std::log1p(1.0f));
        context.near(cached.transmittance(2*ell), std::exp(-2*rate), 1e-14,
                     "exported zero-logit network has constant softplus hazard");
        context.near(renewalTransmittance(model, mean, start, ell, 0.11, 0.51),
                     cached.transmittance(0.11, 0.51), 1e-14, "streaming visibility equals cached visibility");
        Random first(7398), second(7398);
        for (int i = 0; i < 40; ++i) {
            const auto a = cached.sample(first, 0.11, 0.51);
            const auto b = sampleRenewalDistance(model, mean, start, ell, second, 0.11, 0.51);
            context.require(a.hit == b.hit, "cached and streaming draws agree on censoring");
            context.near(a.distance, b.distance, 1e-14, "cached and streaming draws share the same cumulative law");
        }
    }
    context.require(rejects([&] { model.initialize(mean.segments().front(), {RayStartMode::SurfaceOutward, 0}); }),
                    "surface model start must point outward");
    // Nonconstant weights make state resets and changed segment features visible.
    auto varying = constantBundle();
    int parameter = 0;
    for (auto& weight : varying["weights"].items())
        for (auto& value : weight.value()["values"])
            value = static_cast<float>(0.3*std::sin(++parameter*0.17));
    { std::ofstream out(path); out << varying; }
    const auto recurrent = RenewalHazardModel::load(path);
    for (const RayStartCondition start : {RayStartCondition{}, RayStartCondition{RayStartMode::SurfaceOutward, 0.7}}) {
        const auto cached = RenewalRayDistribution::fromModel(recurrent, mean, start, ell);
        for (const auto bounds : {std::array<double,2>{0, 0.79}, {0.113, 0.517}, {0.417, 0.441}, {0.5, 0.8}}) {
            context.near(renewalTransmittance(recurrent, mean, start, ell, bounds[0], bounds[1]),
                cached.transmittance(bounds[0], bounds[1]), 1e-14, "streaming GRU retains the original survival prefix");
            Random first(19317), second(19317);
            for (int i = 0; i < 40; ++i) {
                const auto a = cached.sample(first, bounds[0], bounds[1]);
                const auto b = sampleRenewalDistance(recurrent, mean, start, ell, second, bounds[0], bounds[1]);
                context.require(a.hit == b.hit && a.segment == b.segment, "streaming GRU selects the same hit segment");
                context.near(a.distance, b.distance, 1e-14, "streaming GRU returns the cached inverse distance");
            }
        }
    }
    bundle["kernel"]["parameterization"] = "legacy_sqrt3";
    { std::ofstream out(path); out << bundle; }
    context.require(rejects([&] { RenewalHazardModel::load(path); }), "reject legacy kernel export");
    bundle = constantBundle();
    bundle["weights"]["gru.weight_hh_l0"]["shape"] = {2, 6};
    { std::ofstream out(path); out << bundle; }
    context.require(rejects([&] { RenewalHazardModel::load(path); }), "reject transposed GRU tensor shape");

    Random normalRng(917362);
    for (double mu : {-10.0, -1.0, 0.0, 3.0}) {
        std::array<int,3> counts{};
        constexpr int n = 20000;
        bool positive = true;
        for (int i = 0; i < n; ++i) {
            const double w = samplePositiveNormal(mu, 0.7, normalRng);
            positive &= w > 0 && std::isfinite(w);
            for (int j = 0; j < 3; ++j) counts[j] += w <= std::array<double,3>{0.02,0.3,3.0}[j];
        }
        context.require(positive, "truncated Gaussian samples are finite and strictly positive");
        for (int j = 0; j < 3; ++j) {
            const double w = std::array<double,3>{0.02,0.3,3.0}[j];
            const double cdf = -std::expm1(normalLogCdf((mu-w)/0.7)-normalLogCdf(mu/0.7));
            context.near(counts[j]/static_cast<double>(n), cdf, 6*std::sqrt(cdf*(1-cdf)/n)+0.001,
                         "truncated Gaussian sampling agrees with its normalized CDF");
        }
    }
    double tailMean = 0;
    for (int i = 0; i < 20000; ++i) tailMean += samplePositiveNormal(-1e12, 0.7, normalRng);
    context.near(tailMean/20000/(0.49/1e12), 1, 0.04, "extreme negative tail samples excess without cancellation");

    const auto rotation = Eigen::AngleAxisd(0.7, normalizedOrThrow(Vector3(1,2,3))).toRotationMatrix();
    const auto isotropic = CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::Matern32, 0.1, Vector3::Constant(0.2));
    const auto anisotropic = CovarianceKernel::fromCorrelationLengths(CovarianceKernelType::Matern32, 0.1, Vector3(0.17,0.25,0.4), rotation);
    const Vector3 w = normalizedOrThrow(Vector3(1,-2,3));
    const Vector3 meanG(0.2, -0.3, 0.5), birthMean(-0.1,0.4,0.2), birthG(0.5,-0.7,1.3);
    for (const auto& kernel : {isotropic, anisotropic}) for (bool surface : {false,true}) {
        const std::optional<Vector3> known = surface ? std::optional<Vector3>(birthG) : std::nullopt;
        const auto C = kernel.gradientCovarianceAtZero();
        const Vector3 cw = C*w, a = cw/w.dot(cw);
        const Matrix3 expectedCov = (C-cw*a.transpose())*(surface ? -std::expm1(-0.6) : 1.0);
        Vector3 expectedMean = meanG+a*(-0.4-w.dot(meanG));
        if (surface) expectedMean += std::exp(-0.3)*(birthG-birthMean-a*w.dot(birthG-birthMean));
        Vector3 average = Vector3::Zero();
        Matrix3 second = Matrix3::Zero();
        double projectionError = 0;
        constexpr int n = 20000;
        for (int i = 0; i < n; ++i) {
            const Vector3 g = sampleRenewalGradient(kernel, w, 0.3, -0.4, meanG, birthMean, known, normalRng);
            average += g;
            second += (g-expectedMean)*(g-expectedMean).transpose();
            projectionError = std::max(projectionError, std::abs(g.dot(w)+0.4));
        }
        context.near(projectionError, 0, 2e-15, "full gradient preserves the sampled physical directional derivative");
        context.near((average/n-expectedMean).norm(), 0, 0.018, "Renewal A/B transverse gradient mean");
        context.near((second/n-expectedCov).norm(), 0, 0.018, "Renewal A/B transverse gradient covariance");
    }

    { std::ofstream out(path); out << constantBundle(true); }
    const auto fullModel = RenewalHazardModel::load(path);
    context.require(fullModel.hasMixture() && !model.hasMixture(), "hazard bundles remain usable but cannot sample normals");
    ExperimentConfig render;
    render.field = {std::make_shared<ConstantMean>(0), isotropic,
        {Point3::Constant(-0.15), Point3::Constant(0.15)}};
    render.transportMode = "neural_renewal";
    render.renewal.model = std::make_shared<const RenewalHazardModel>(fullModel);
    render.render.width = render.render.height = 4;
    render.render.samplesPerPixel = 16;
    render.render.cameraPosition = Point3(0,0,0.5);
    render.render.cameraTarget = Point3::Zero();
    render.render.environment = "unit_white";
    render.render.rouletteStartDepth = 1000;
    render.render.safetyDepthCap = 128;
    render.material.conductor.forceUnitFresnel = true;
    const RenewalMedium neural(render.field, fullModel);
    const RenewalMedium pointMedium(render.field,fullModel,0.25,"point_linear");
    context.require(rejects([&] { RenewalMedium(render.field,fullModel,0.25,"invalid"); }),
                    "invalid point profile mode rejected");
    context.require(rejects([&] { RenewalMedium(render.field, model); }), "normal renderer rejects hazard-only model");
    const Ray ray{Point3(0,0,0.15), -Vector3::UnitZ()};
    const auto flight = neural.beginFlight(ray);
    const auto pointFlight = pointMedium.beginFlight(ray);
    context.require(pointFlight && pointFlight->mean.constructedSegmentCount() == 1,
                    "point flight initializes without constructing its entire ray");
    context.near(pointMedium.transmittance(*pointFlight,0.01),neural.transmittance(*flight,0.01),1e-12,
                 "point visibility preserves the constant-hazard law");
    context.require(pointFlight->mean.constructedSegmentCount() == 1,
                    "short visibility does not construct unvisited future segments");
    Random pointSampleRng(319), cubicSampleRng(319);
    const auto pointSample = pointMedium.sample(*pointFlight,pointSampleRng);
    const auto cubicSample = neural.sample(*flight,cubicSampleRng);
    context.require(pointSample.hit == cubicSample.hit,"point and cubic constant-field hit agreement");
    context.near(pointSample.distance,cubicSample.distance,1e-10,"point sampling uses cumulative inversion");
    context.require(flight.has_value(), "neural external ray enters the field");
    const auto law = RenewalRayDistribution::fromModel(fullModel, flight->mean, flight->start, flight->ell);
    context.near(neural.transmittance(*flight, 0.213), law.transmittance(0.213), 1e-14,
                 "neural medium visibility uses the identical cumulative law");
    Random hitRng(531), distanceRng(531);
    const auto draw = neural.sample(*flight, hitRng);
    const auto distance = law.sample(distanceRng);
    context.require(draw.hit == distance.hit, "surface sampler preserves hazard-only hit probability");
    if (draw.hit) {
        context.near(draw.distance, distance.distance, 1e-12, "surface sampler preserves hazard-only sampled distance");
        context.near(draw.gradient.dot(ray.direction), -flight->sigma/flight->ell*draw.speed, 1e-14, "speed rescales to physical gradient");
        const Ray next{draw.position, normalizedOrThrow(reflectTravelDirection(ray.direction, draw.normal))};
        const auto continued = neural.beginFlight(next, draw.gradient);
        context.require(continued && continued->start.mode == RayStartMode::SurfaceOutward, "reflection starts mode B");
        context.near(continued->start.outwardDerivative, continued->ell/continued->sigma*next.direction.dot(draw.gradient),
                     1e-14, "B initialization retains gradient magnitude");
        context.near(neural.surfaceTransmittance(next, draw.gradient, 0.1), neural.transmittance(*continued, 0.1),
                     1e-14, "shadow direction uses the same surface gradient and a fresh B flight");
        context.near(neural.surfaceTransmittance({draw.position,-next.direction}, draw.gradient, 0.1), 0, 0,
                     "inward shadow direction is blocked");
    }
    render.render.threadCount = 1;
    const auto serial = renderAnalyticScene(render);
    render.render.threadCount = 3;
    const auto parallel = renderAnalyticScene(render);
    context.require(serial.statistics.realCollisions > 0 && serial.statistics.renewal.flights > serial.statistics.paths,
                    "neural path tracer performs multi-bounce transport");
    context.require(serial.statistics.safetyCapTerminations == 0 && serial.statistics.numericalFailures == 0,
                    "neural energy test has no capped or failed paths");
    context.require(serial.statistics.renewal.mixtureQueries == serial.statistics.realCollisions,
                    "mixture head runs exactly once at each real collision");
    context.require(serial.statistics.realCollisions == parallel.statistics.realCollisions &&
                    serial.statistics.renewal.segments == parallel.statistics.renewal.segments,
                    "neural trajectory statistics are independent of worker count");
    for (std::size_t i = 0; i < serial.pixels.size(); ++i) {
        context.near((serial.pixels[i]-Spectrum::Ones()).norm(), 0, 1e-14, "unit Fresnel white furnace conserves energy");
        context.near((serial.pixels[i]-parallel.pixels[i]).norm(), 0, 0, "neural image is independent of worker count");
    }
    render.render.environment = "directional_gradient";
    render.material.conductor.forceUnitFresnel = false;
    render.render.threadCount = 1;
    const auto conductorSerial = renderAnalyticScene(render);
    render.render.threadCount = 3;
    const auto conductorParallel = renderAnalyticScene(render);
    bool nontrivial = false;
    for (std::size_t i = 0; i < conductorSerial.pixels.size(); ++i) {
        nontrivial |= (conductorSerial.pixels[i]-Spectrum::Ones()).norm() > 0.1;
        context.require(conductorSerial.pixels[i].allFinite() && (conductorSerial.pixels[i].array() >= 0).all(),
                        "neural conductor rendering is finite and nonnegative");
        context.near((conductorSerial.pixels[i]-conductorParallel.pixels[i]).norm(), 0, 0,
                     "Fresnel and directional neural image is identical across worker counts");
    }
    context.require(nontrivial, "neural render evaluates conductor Fresnel and directional illumination");

    std::vector<std::string> backends;
    if (renewalTorchAvailable()) backends.push_back("torch_cpu");
    if (renewalCudaAvailable()) backends.push_back("torch_cuda");
    context.require(resolveRenewalBackend("scalar") == "scalar", "scalar inference stays selectable");
    context.require(rejects([] { resolveRenewalBackend("unknown"); }), "unknown inference backend is rejected");
    if (!renewalTorchAvailable()) {
        context.require(resolveRenewalBackend("auto") == "scalar", "auto falls back without LibTorch");
        context.require(rejects([] { resolveRenewalBackend("torch_cpu"); }), "explicit missing LibTorch is rejected");
    }
    auto batchBundle = constantBundle(true);
    parameter = 0;
    for (auto& weight : batchBundle["weights"].items())
        for (auto& value : weight.value()["values"])
            value = static_cast<float>(0.3*std::sin(++parameter*0.17));
    { std::ofstream out(path); out << batchBundle; }
    const auto batchModel = RenewalHazardModel::load(path);
    for (const auto& backend : backends) {
        RenewalBatchSession batch(batchModel,7,backend);
        std::vector<int> ids{5,0,6,2,1};
        std::vector<RayMeanSegment> first(ids.size(),mean.segments().front());
        std::vector<RayStartCondition> starts(ids.size());
        starts[1] = {RayStartMode::SurfaceOutward,0.71};
        starts[3] = {RayStartMode::SurfaceOutward,0.24};
        context.require(rejects([&] { batch.evaluate(ids,first); }), "batch rejects uninitialized slots");
        batch.initialize(ids,first,starts);
        std::vector<RenewalHazardModel::State> states(7), entering(7);
        std::vector<RayMeanSegment> last(7);
        for (std::size_t i = 0; i < ids.size(); ++i) states[ids[i]] = batchModel.initialize(first[i],starts[i]);
        context.require(rejects([&] { batch.mixture({5},{0.5}); }), "mixture requires an evaluated segment");
        for (int step = 0; step < 19; ++step) {
            std::vector<int> active = step%2 ? std::vector<int>{2,0,5} : ids;
            if (step == 9) {
                const RayStartCondition reset{RayStartMode::SurfaceOutward,0.91};
                batch.initialize({2},{first[0]},{reset});
                states[2] = batchModel.initialize(first[0],reset);
                context.require(rejects([&] { batch.mixture({2},{0.2}); }), "slot reset invalidates old mixture context");
            }
            std::vector<RayMeanSegment> segments;
            for (int id : active) {
                const double begin = 0.13*step, width = 0.07+0.02*id;
                segments.push_back({begin,begin+width,{0.03*id,-0.07,0.11*(id+1),0.2-0.013*step}});
            }
            context.require(rejects([&] { batch.evaluate({5,5},{segments[0],segments[0]}); }),
                            "duplicate batch slots cannot advance a ray twice");
            const auto rates = batch.evaluate(active,segments);
            for (std::size_t i = 0; i < active.size(); ++i) {
                const int id = active[i];
                entering[id] = states[id]; last[id] = segments[i];
                const auto reference = batchModel.evaluate(states[id],segments[i]);
                states[id] = reference.nextState;
                for (int j = 0; j < 4; ++j) context.near(rates[i][j],reference.rates[j],3e-6,
                    backend+" retains independent A/B recurrent states in reordered sparse batches");
            }
            // Also query slots absent from this wave: their cached context must survive.
            const std::vector<double> coordinates{0,0.13,0.8,1,0.47};
            const auto mixtures = batch.mixture(ids,coordinates);
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto reference = batchModel.mixture(entering[ids[i]],last[ids[i]],coordinates[i]);
                for (std::size_t j = 0; j < reference.weights.size(); ++j) {
                    context.near(mixtures[i].weights[j],reference.weights[j],3e-6,backend+" mixture weights parity");
                    context.near(mixtures[i].means[j],reference.means[j],3e-6,backend+" mixture entering-state means parity");
                    context.near(mixtures[i].scales[j],reference.scales[j],3e-6,backend+" mixture scales parity");
                }
            }
        }
        context.require(batch.evaluate({},{}).empty() && batch.mixture({},{}).empty(), "empty batches need no inference");
        context.require(rejects([&] { batch.mixture({0},{1.1}); }), "invalid batch mixture coordinate rejected");

        // Raw inputs must reach asinh as doubles, not overflow during an early
        // float cast. Exercise initial, segment and mixture transforms together.
        const std::vector<int> wideIds{1,0};
        const std::vector<RayMeanSegment> wideSegments{
            {0,0.125,{0,0,1e100,-2e100}}, {0,0.5,{0,0,-1e120,3e120}}};
        const std::vector<RayStartCondition> wideStarts{
            {}, {RayStartMode::SurfaceOutward,0.7}};
        RenewalBatchSession wideBatch(batchModel,2,backend);
        wideBatch.initialize(wideIds,wideSegments,wideStarts);
        const auto wideRates = wideBatch.evaluate(wideIds,wideSegments);
        const std::vector<double> wideU{0.23,0.87};
        const auto wideMixtures = wideBatch.mixture(wideIds,wideU);
        for (std::size_t i = 0; i < wideIds.size(); ++i) {
            const auto initial = batchModel.initialize(wideSegments[i],wideStarts[i]);
            const auto reference = batchModel.evaluate(initial,wideSegments[i]);
            const auto mix = batchModel.mixture(initial,wideSegments[i],wideU[i]);
            for (int j = 0; j < 4; ++j)
                context.relative(wideRates[i][j],reference.rates[j],3e-6,
                    backend+" transforms double-range features before float conversion");
            for (std::size_t j = 0; j < mix.weights.size(); ++j) {
                context.near(wideMixtures[i].weights[j],mix.weights[j],3e-6,
                    backend+" mixture transforms double-range b/db but leaves u unchanged");
                context.relative(wideMixtures[i].means[j],mix.means[j],3e-6,
                    backend+" retains double mixture derivative correction");
                context.relative(wideMixtures[i].scales[j],mix.scales[j],3e-6,
                    backend+" double-range mixture scale parity");
            }
        }
        auto invalidSegment = wideSegments[0];
        invalidSegment.polynomial.d = std::numeric_limits<double>::infinity();
        context.require(rejects([&] { wideBatch.evaluate({1},{invalidSegment}); }),
            backend+" rejects nonfinite raw inputs before device dispatch");

        // Cover softplus underflow, the threshold=20 branch, and values whose
        // naive exp would overflow. The scalar model remains the reference.
        for (const std::array<float,4> logits : {
                 std::array<float,4>{-100,-0.2f,20,100}, {-20,0,19.9f,20.1f}}) {
            auto activationBundle = constantBundle(true);
            activationBundle["weights"]["hazard_head.4.bias"]["values"] = logits;
            activationBundle["weights"]["mixture_head.4.bias"]["values"] =
                std::vector<float>{0,0,0,0,logits[0],logits[3]};
            { std::ofstream out(path); out << activationBundle; }
            const auto activationModel = RenewalHazardModel::load(path);
            RenewalBatchSession activated(activationModel,1,backend);
            const auto& firstSegment = mean.segment(0);
            activated.initialize({0},{firstSegment},{{}});
            const auto rates = activated.evaluate({0},{firstSegment});
            const auto mixes = activated.mixture({0},{0.41});
            const auto initial = activationModel.initialize(firstSegment,{});
            const auto reference = activationModel.evaluate(initial,firstSegment);
            const auto mix = activationModel.mixture(initial,firstSegment,0.41);
            for (int j = 0; j < 4; ++j)
                context.relative(rates[0][j],reference.rates[j],3e-6,
                    backend+" softplus retains stable tails and threshold semantics");
            for (std::size_t j = 0; j < mix.scales.size(); ++j)
                context.relative(mixes[0].scales[j],mix.scales[j],3e-6,
                    backend+" mixture applies softplus and sigma floor exactly once");
        }
        auto overflowBundle = constantBundle(true);
        overflowBundle["weights"]["mixture_head.2.bias"]["values"] = std::vector<float>(4,2);
        overflowBundle["weights"]["mixture_head.4.weight"]["values"][4*4] = -3e38f;
        { std::ofstream out(path); out << overflowBundle; }
        const auto overflowModel = RenewalHazardModel::load(path);
        RenewalBatchSession overflowBatch(overflowModel,1,backend);
        overflowBatch.initialize({0},{mean.segment(0)},{{}});
        overflowBatch.evaluate({0},{mean.segment(0)});
        context.require(rejects([&] { overflowBatch.mixture({0},{0.3}); }),
            backend+" softplus must not hide an overflowed negative-infinite scale logit");

        render.renewal.backend = backend;
        for (int size : {3,16}) {
            render.renewal.batchSize = size;
            std::atomic<std::uint64_t> progress{0};
            const auto batched = renderAnalyticScene(render,&progress);
            context.require(batched.statistics.paths == conductorSerial.statistics.paths &&
                progress.load() == conductorSerial.statistics.paths, "wavefront completes every camera sample and reports progress");
            context.require(batched.statistics.renewal.maximumBatchSize <= static_cast<std::uint64_t>(size) &&
                batched.statistics.renewal.segmentBatches > 0, "wavefront bounds and records its inference batches");
            context.require(batched.statistics.renewal.mixtureQueries == batched.statistics.realCollisions,
                "wavefront queries mixtures only on hits");
            for (std::size_t i = 0; i < batched.pixels.size(); ++i)
                context.near((batched.pixels[i]-conductorSerial.pixels[i]).norm(),0,3e-5,
                    backend+" wavefront keeps per-pixel random streams, Fresnel and accumulation order");
        }
        render.material.conductor.forceUnitFresnel = true;
        render.render.environment = "unit_white";
        const auto white = renderAnalyticScene(render);
        for (const auto& pixel : white.pixels)
            context.near((pixel-Spectrum::Ones()).norm(),0,1e-14,backend+" wavefront white furnace conserves energy");
        render.material.conductor.forceUnitFresnel = false;
        render.render.environment = "directional_gradient";
        render.render.rouletteStartDepth = 1;
        render.render.safetyDepthCap = 1;
        render.renewal.backend = "scalar";
        const auto cappedScalar = renderAnalyticScene(render);
        render.renewal.backend = backend;
        const auto cappedBatch = renderAnalyticScene(render);
        context.require(cappedBatch.statistics.safetyCapTerminations > 0 && cappedBatch.statistics.rouletteTerminations > 0,
            "wavefront exercises depth cap and Russian roulette");
        context.require(cappedBatch.statistics.safetyCapTerminations == cappedScalar.statistics.safetyCapTerminations &&
            cappedBatch.statistics.rouletteTerminations == cappedScalar.statistics.rouletteTerminations &&
            cappedBatch.statistics.escapedPaths == cappedScalar.statistics.escapedPaths,
            "wavefront termination reasons match scalar transport");
        for (std::size_t i = 0; i < cappedBatch.pixels.size(); ++i)
            context.near((cappedBatch.pixels[i]-cappedScalar.pixels[i]).norm(),0,3e-5,
                backend+" preserves capped and roulette path contributions");
        render.render.rouletteStartDepth = 1000;
        render.render.safetyDepthCap = 128;
        render.renewal.profileMode = "point_linear";
        render.renewal.backend = "scalar";
        const auto pointScalar = renderAnalyticScene(render);
        render.renewal.backend = backend;
        const auto pointBatch = renderAnalyticScene(render);
        context.require(pointBatch.statistics.renewal.segments == pointScalar.statistics.renewal.segments &&
            pointBatch.statistics.realCollisions == pointScalar.statistics.realCollisions,
            backend+" point mode preserves scalar/batch path statistics");
        for (std::size_t i = 0; i < pointBatch.pixels.size(); ++i)
            context.near((pointBatch.pixels[i]-pointScalar.pixels[i]).norm(),0,3e-5,
                         backend+" point mode supports complete neural path tracing");
        render.renewal.profileMode = "cubic";
    }
    std::filesystem::remove(path);
}

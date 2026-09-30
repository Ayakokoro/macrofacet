#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/gpss/ShaderBallMean.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mf {
namespace {

Vector3 vector3(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) throw std::invalid_argument("expected a 3-vector");
    return Vector3(value[0].get<double>(), value[1].get<double>(), value[2].get<double>());
}

Matrix3 matrix3(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) throw std::invalid_argument("expected a 3x3 matrix");
    Matrix3 result;
    for (int i = 0; i < 3; ++i) {
        if (!value[i].is_array() || value[i].size() != 3) {
            throw std::invalid_argument("expected a 3x3 matrix");
        }
        for (int j = 0; j < 3; ++j) result(i, j) = value[i][j].get<double>();
    }
    return result;
}

nlohmann::json toArray(const Vector3& v) { return {v.x(), v.y(), v.z()}; }

void requirePositiveFinite(double value, const char* name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be a finite positive number");
    }
}

ConditionalBirthConfig parseConditionalBirth(const nlohmann::json& value,
                                              const std::string& location) {
    if (!value.is_object())
        throw std::invalid_argument(location + " must be an object");
    ConditionalBirthConfig result;
    result.policy = value.value("policy", result.policy);
    if (result.policy == "sample_positive_exterior") {
        return result;
    }
    if (result.policy != "fixed_observation") {
        throw std::invalid_argument(
            location + ".policy must be sample_positive_exterior or fixed_observation");
    }
    if (!value.contains("value") || !value.contains("gradient")) {
        throw std::invalid_argument(
            location + " fixed_observation requires value and gradient");
    }
    result.value = value.at("value").get<double>();
    result.gradient = vector3(value.at("gradient"));
    if (!(result.value >= 0.0) || !std::isfinite(result.value) ||
        !result.gradient.allFinite()) {
        throw std::invalid_argument(location + " fixed observation must be finite with value >= 0");
    }
    return result;
}

bool isTransmittanceMode(const std::string& mode) {
    return mode == "classic_local" || mode == "classic_global" ||
           mode == "global_conditional";
}

nlohmann::json conditionalBirthJson(const ConditionalBirthConfig& birth) {
    nlohmann::json result{{"policy", birth.policy}};
    if (birth.policy == "fixed_observation") {
        result["value"] = birth.value;
        result["gradient"] = toArray(birth.gradient);
    }
    return result;
}

SquaredExponentialKernel roughnessKernel(double sigma, double roughness, NdfFamily family) {
    requirePositiveFinite(sigma, "sigma");
    requirePositiveFinite(roughness, "material roughness");
    if (family != NdfFamily::GeneralizedGaussian) {
        throw std::invalid_argument("material roughness requires the generalized_gaussian NDF family");
    }
    const double valueVariance = sigma * sigma;
    // Beckmann alpha is sqrt(2) times the standard deviation of each slope
    // component when the mean height gradient has unit length.
    const double gradientStddev = roughness / std::sqrt(2.0);
    const double gradientVariance = gradientStddev * gradientStddev;
    const double correlationLength = sigma / gradientStddev;
    if (!(valueVariance > 0.0) || !std::isfinite(valueVariance) ||
        !(gradientVariance > 0.0) || !std::isfinite(gradientVariance) ||
        !(correlationLength > 0.0) || !std::isfinite(correlationLength)) {
        throw std::invalid_argument(
            "roughness and sigma must yield finite positive variances and correlation length");
    }
    const double inverseLength = gradientStddev / sigma;
    const double precision = inverseLength * inverseLength;
    const double actualGradientVariance = valueVariance * precision;
    if (!(precision > 0.0) || !std::isfinite(precision) ||
        !(actualGradientVariance > 0.0) || !std::isfinite(actualGradientVariance)) {
        throw std::invalid_argument(
            "roughness / (sqrt(2) * sigma) must yield finite positive kernel precision and gradient covariance");
    }
    return SquaredExponentialKernel(sigma, precision * Matrix3::Identity());
}

} // namespace

GPSSField buildDefaultField() {
    GPSSField field{
        std::make_shared<PlaneMean>(Vector3::UnitZ(), 0.0),
        SquaredExponentialKernel::fromCorrelationLengths(0.1, Vector3(0.2, 0.2, 0.2)),
        {Point3(-2.0, -2.0, -0.3), Point3(2.0, 2.0, 0.3)}};
    field.validate();
    return field;
}

void requireNanoVdbField(const ExperimentConfig& config) {
    if (!config.mediumDensity || !config.densityMajorantGrid || !config.field.mean ||
        std::string(config.field.mean->typeName()) != "nanovdb" ||
        !(config.field.mean->voxelSizeHint() > 0.0)) {
        throw std::invalid_argument("experiment tracing requires a prepared NanoVDB field");
    }
}

void applyFieldOverrides(ExperimentConfig& config, std::optional<double> sigma,
                         std::optional<double> roughness, bool preserveSlope) {
    config.preparedAreaMajorant.reset();
    if (sigma) requirePositiveFinite(*sigma, "sigma");
    // The CLI is the other way a sigma can reach a baked field, so it obeys the
    // same rule as the config key: sigma is part of the data, not a knob.
    // Sweeping it would reinterpret the band (+-3 sigma) and the background
    // at the wrong scale and render a plausible but wrong surface.
    // An agreeing value stays allowed -- it keeps a sweep script that passes one
    // sigma to every config working -- but the field's own spelling wins, so the
    // override really is the no-op it claims to be rather than a swap to a value
    // one float32 ulp away.
    if (sigma && config.field.mean) {
        if (const std::optional<double> baked = config.field.mean->intrinsicSigma()) {
            // Compared as float32 because that is the precision a baked field
            // holds sigma at, so it is the finest distinction that means
            // anything here; anything further apart is a different bake.
            if (static_cast<float>(*baked) != static_cast<float>(*sigma)) {
                throw std::invalid_argument(
                    "--sigma (" + std::to_string(*sigma) + ") disagrees with the baked field (" +
                    std::to_string(*baked) + "); re-bake instead of overriding");
            }
            sigma = *baked;
        }
    }
    // Explicit CLI roughness replaces either config roughness or legacy
    // anisotropic lengths. Config roughness also stays fixed as sigma changes.
    const std::optional<double> effectiveRoughness = roughness ? roughness : config.materialRoughness;
    if (effectiveRoughness) {
        config.field.kernel = roughnessKernel(sigma.value_or(config.field.kernel.sigma()),
                                             *effectiveRoughness, config.material.ndfFamily);
        config.materialRoughness = effectiveRoughness;
        config.field.validate();
    } else if (sigma) {
        const double oldSigma = config.field.kernel.sigma();
        Matrix3 precision = config.field.kernel.precision();
        if (preserveSlope) {
            const double scale = oldSigma / *sigma;
            precision *= scale * scale;
        }
        config.field.kernel = SquaredExponentialKernel(*sigma, precision);
        config.field.validate();
    }
}

ExperimentConfig loadExperimentConfig(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open experiment config: " + path.string());
    nlohmann::json root;
    stream >> root;
    ExperimentConfig config;
    config.schemaVersion = root.value("schema_version", 1);
    if (config.schemaVersion != 1) throw std::invalid_argument("unsupported config schema version");
    config.seed = root.value("seed", config.seed);
    for (const char* obsolete : {"modes", "reference", "fixed_flight"}) {
        if (root.contains(obsolete))
            throw std::invalid_argument(std::string("obsolete configuration block: ") + obsolete);
    }

    const auto& fieldJson = root.at("field");
    config.sourceFieldSpec = fieldJson.dump();
    if (fieldJson.contains("bake_voxel_size")) {
        config.bakeVoxelSize = fieldJson.at("bake_voxel_size").get<double>();
        requirePositiveFinite(*config.bakeVoxelSize, "field.bake_voxel_size");
    }
    // The registry owns the procedural types (plane / sphere / cutaway_sphere /
    // shader_ball / constant) and macrofacet_field adds "nanovdb" at startup.
    // Keeping the lookup here means the core library never learns about NanoVDB.
    MeanBuildResult built = buildMeanFromJson(fieldJson);
    // The field schema splits in two here, and the split is the field's to
    // declare (MeanField::intrinsicSigma), not the config's to guess: this
    // library never learns which mean types are baked. A procedural mean is
    // scale-free, so "sigma" is required and is the only source. A baked mean
    // froze sigma into its own data, so "sigma" may be omitted and the file
    // supplies it -- and when it is given anyway the type's builder has already
    // cross-checked it against the file (NanoVdbMean::open rejects a
    // disagreement) rather than letting the config silently win.
    const std::optional<double> fromField = built.mean->intrinsicSigma();
    if (!fieldJson.contains("sigma") && !fromField) {
        throw std::invalid_argument(
            std::string("field.sigma is required for mean_type '") + built.mean->typeName() +
            "', which fixes no scale of its own");
    }
    const double sigma = fieldJson.contains("sigma")
        ? fieldJson.at("sigma").get<double>() : *fromField;
    const auto& material = root.at("material");
    // Accept the historical field key while writing new configs under material.
    const std::string family = material.contains("ndf_family")
        ? material.at("ndf_family").get<std::string>()
        : fieldJson.at("ndf_family").get<std::string>();
    NdfFamily ndfFamily;
    if (family == "generalized_gaussian") ndfFamily = NdfFamily::GeneralizedGaussian;
    else if (family == "beckmann_limit") ndfFamily = NdfFamily::BeckmannLimit;
    else if (family == "ggx") ndfFamily = NdfFamily::GGXBaseline;
    else throw std::invalid_argument("unknown NDF family: " + family);
    SquaredExponentialKernel kernel;
    if (material.contains("roughness")) {
        if (fieldJson.contains("correlation_lengths")) {
            throw std::invalid_argument("material.roughness and field.correlation_lengths are mutually exclusive");
        }
        if (!material.at("roughness").is_number()) {
            throw std::invalid_argument("material.roughness must be a finite positive number");
        }
        config.materialRoughness = material.at("roughness").get<double>();
        kernel = roughnessKernel(sigma, *config.materialRoughness, ndfFamily);
    } else {
        Vector3 lengths;
        const auto& lengthJson = fieldJson.at("correlation_lengths");
        if (!lengthJson.is_array() || lengthJson.size() != 3) {
            throw std::invalid_argument("correlation_lengths must contain three entries");
        }
        for (int i = 0; i < 3; ++i) {
            lengths[i] = lengthJson[i].is_null() ? std::numeric_limits<double>::infinity()
                                                 : lengthJson[i].get<double>();
        }
        const Matrix3 rotation = fieldJson.contains("kernel_rotation") ?
            matrix3(fieldJson.at("kernel_rotation")) : Matrix3::Identity();
        kernel = SquaredExponentialKernel::fromCorrelationLengths(sigma, lengths, rotation);
    }
    // A baked grid fixes its own spatial coverage, just as it fixes sigma.
    // Legacy domain_min/domain_max entries are ignored for baked fields so a
    // config cannot silently clip a newly imported mesh at an old bbox.
    Bounds3 domain;
    if (built.activeDomain) {
        domain = *built.activeDomain;
    } else {
        if (!fieldJson.contains("domain_min") || !fieldJson.contains("domain_max"))
            throw std::invalid_argument("procedural field requires domain_min and domain_max");
        domain = {vector3(fieldJson.at("domain_min")),
                  vector3(fieldJson.at("domain_max"))};
    }
    config.field = {std::move(built.mean), kernel, domain};
    config.material.ndfFamily = ndfFamily;
    const std::string gpModel = material.value("gp_model", "local_tangent");
    if (gpModel == "local_tangent") {
        config.material.gpModel = GpModel::LocalTangent;
    } else if (gpModel == "global_pointwise") {
        config.material.gpModel = GpModel::GlobalPointwise;
        if (fieldJson.at("mean_type") == "nanovdb" &&
            fieldJson.value("use_alpha_grid", true)) {
            throw std::invalid_argument(
                "global_pointwise requires field.use_alpha_grid=false");
        }
    } else {
        throw std::invalid_argument("unknown material.gp_model: " + gpModel);
    }
    const auto alpha = material.contains("ggx_alpha") ? material.at("ggx_alpha") :
        fieldJson.value("ggx_alpha", nlohmann::json());
    if (!alpha.is_null()) {
        config.material.ggxAlpha = Vector2(alpha[0].get<double>(), alpha[1].get<double>());
    }
    config.material.alphaField = std::move(built.alphaField);
    config.material.alphaPositiveOnDensitySupport = built.alphaPositiveOnDensitySupport;

    config.material.conductor.eta = vector3(material.at("eta_rgb"));
    config.material.conductor.k = vector3(material.at("k_rgb"));
    config.material.conductor.forceUnitFresnel =
        material.value("force_unit_fresnel_for_energy_test", false);
    const auto& transport = root.at("transport");
    config.transportMode = transport.value("mode", "classic");
    if (config.transportMode != "classic" && config.transportMode != "classic_local" &&
        config.transportMode != "classic_global" &&
        config.transportMode != "global_conditional" && config.transportMode != "all")
        throw std::invalid_argument("unknown transport.mode: " + config.transportMode);
    for (const char* obsolete : {"conditional29", "correlated_sampler", "external_policy",
                                 "hard_depth_cap", "next_event_estimation"}) {
        if (transport.contains(obsolete))
            throw std::invalid_argument(std::string("obsolete transport setting: ") + obsolete);
    }
    config.beckmannMixtureWeight = transport.value("beckmann_mixture_weight", 0.5);
    config.classicPhaseProposal = transport.value("classic_phase_proposal", "uniform");
    if (config.classicPhaseProposal != "uniform" &&
        config.classicPhaseProposal != "paper_mixture" &&
        config.classicPhaseProposal != "target_vndf") {
        throw std::invalid_argument("unknown classic phase proposal");
    }
    if (config.classicPhaseProposal == "target_vndf" &&
        config.material.ndfFamily == NdfFamily::GGXBaseline) {
        throw std::invalid_argument("target_vndf requires a Gaussian NDF");
    }
    config.render.rouletteStartDepth = transport.value("roulette_start_depth", 5);

    const auto& numeric = root.at("numeric");
    config.numeric.relativeTolerance = numeric.at("relative_tolerance").get<double>();
    config.numeric.absoluteTolerance = numeric.at("absolute_tolerance").get<double>();
    config.numeric.maxQuadratureSubdivisions = numeric.at("max_quadrature_subdivisions").get<int>();
    config.numeric.maxRootIterations = numeric.at("max_root_iterations").get<int>();
    config.numeric.distanceAbsoluteTolerance = numeric.value("distance_absolute_tolerance", 1e-10);
    config.numeric.distanceRelativeTolerance = numeric.value("distance_relative_tolerance", 1e-8);
    if (numeric.value("allow_unreported_jitter", false)) {
        throw std::invalid_argument("unreported covariance jitter is intentionally unsupported");
    }

    const auto& render = root.at("render");
    config.render.width = render.at("width").get<int>();
    config.render.height = render.at("height").get<int>();
    config.render.samplesPerPixel = render.at("samples_per_pixel").get<int>();
    config.render.cameraPosition = vector3(render.at("camera_position"));
    config.render.cameraTarget = vector3(render.at("camera_target"));
    config.render.verticalFovDegrees = render.at("vertical_fov_degrees").get<double>();
    config.render.environment = render.at("environment").get<std::string>();
    if (render.contains("flight_table_cells"))
        throw std::invalid_argument("obsolete render.flight_table_cells setting");
    config.render.threadCount = render.value("thread_count", 0);

    if (root.contains("transmittance")) {
        const auto& experiment = root.at("transmittance");
        if (!experiment.is_object())
            throw std::invalid_argument("transmittance must be an object");
        TransmittanceConfig transmittance;
        if (experiment.contains("modes")) {
            const auto& modes = experiment.at("modes");
            if (!modes.is_array() || modes.empty())
                throw std::invalid_argument("transmittance.modes must be a nonempty array");
            transmittance.modes.clear();
            std::unordered_set<std::string> uniqueModes;
            for (const auto& entry : modes) {
                const std::string mode = entry.get<std::string>();
                if (!isTransmittanceMode(mode))
                    throw std::invalid_argument("unknown transmittance mode: " + mode);
                if (!uniqueModes.insert(mode).second)
                    throw std::invalid_argument("duplicate transmittance mode: " + mode);
                transmittance.modes.push_back(mode);
            }
        }
        if (experiment.contains("curve")) {
            const auto& curve = experiment.at("curve");
            transmittance.bins = curve.value("bins", transmittance.bins);
            transmittance.spacing = curve.value("spacing", transmittance.spacing);
            transmittance.distanceOrigin =
                curve.value("distance_origin", transmittance.distanceOrigin);
        }
        if (transmittance.bins < 2 || transmittance.spacing != "linear" ||
            transmittance.distanceOrigin != "ray_origin") {
            throw std::invalid_argument(
                "transmittance.curve requires bins >= 2, spacing=linear, "
                "and distance_origin=ray_origin");
        }
        if (experiment.contains("monte_carlo")) {
            const auto& monteCarlo = experiment.at("monte_carlo");
            transmittance.trialsPerRay =
                monteCarlo.value("trials_per_ray", transmittance.trialsPerRay);
            transmittance.confidenceLevel =
                monteCarlo.value("confidence_level", transmittance.confidenceLevel);
            transmittance.writeRawSamples =
                monteCarlo.value("write_raw_samples", transmittance.writeRawSamples);
        }
        if (transmittance.trialsPerRay < 1 ||
            !(transmittance.confidenceLevel > 0.0 &&
              transmittance.confidenceLevel < 1.0) ||
            !std::isfinite(transmittance.confidenceLevel)) {
            throw std::invalid_argument("invalid transmittance Monte Carlo settings");
        }
        if (experiment.contains("conditional_birth")) {
            transmittance.conditionalBirth = parseConditionalBirth(
                experiment.at("conditional_birth"), "transmittance.conditional_birth");
        }
        if (experiment.contains("comparison")) {
            const auto& comparison = experiment.at("comparison");
            transmittance.localTransportReference = comparison.value(
                "local_transport_reference", transmittance.localTransportReference);
            if (transmittance.localTransportReference != "classic_global") {
                throw std::invalid_argument(
                    "transmittance comparison currently requires classic_global as the local reference");
            }
            if (comparison.contains("constant_exponential")) {
                const auto& exponential = comparison.at("constant_exponential");
                transmittance.constantExponentialEnabled =
                    exponential.value("enabled", transmittance.constantExponentialEnabled);
                transmittance.constantExponentialFit =
                    exponential.value("fit", transmittance.constantExponentialFit);
            }
            if (transmittance.constantExponentialFit != "censored_mle") {
                throw std::invalid_argument(
                    "transmittance constant exponential fit must be censored_mle");
            }
        }
        if (!experiment.contains("rays") || !experiment.at("rays").is_array() ||
            experiment.at("rays").empty()) {
            throw std::invalid_argument("transmittance.rays must be a nonempty array");
        }
        std::unordered_set<std::string> rayIds;
        for (std::size_t index = 0; index < experiment.at("rays").size(); ++index) {
            const auto& rayJson = experiment.at("rays").at(index);
            const std::string location =
                "transmittance.rays[" + std::to_string(index) + "]";
            TransmittanceRayConfig ray;
            ray.id = rayJson.at("id").get<std::string>();
            ray.group = rayJson.value("group", ray.group);
            ray.origin = vector3(rayJson.at("origin"));
            ray.direction = normalizedOrThrow(vector3(rayJson.at("direction")));
            ray.maximumDistance = rayJson.at("max_distance").get<double>();
            if (ray.id.empty() || !rayIds.insert(ray.id).second)
                throw std::invalid_argument(location + ".id must be nonempty and unique");
            if (ray.group.empty())
                throw std::invalid_argument(location + ".group must be nonempty");
            requirePositiveFinite(ray.maximumDistance,
                                  (location + ".max_distance").c_str());
            if (!ray.origin.allFinite())
                throw std::invalid_argument(location + ".origin must be finite");
            if (rayJson.contains("bins")) {
                ray.bins = rayJson.at("bins").get<int>();
                if (*ray.bins < 2)
                    throw std::invalid_argument(location + ".bins must be at least 2");
            }
            if (rayJson.contains("conditional_birth")) {
                ray.conditionalBirth = parseConditionalBirth(
                    rayJson.at("conditional_birth"), location + ".conditional_birth");
            }
            transmittance.rays.push_back(std::move(ray));
        }
        config.transmittance = std::move(transmittance);
    }
    config.outputDirectory = root.at("output_directory").get<std::string>();

    if (config.render.width < 1 || config.render.height < 1 || config.render.samplesPerPixel < 1 ||
        config.render.threadCount < 0 ||
        config.render.rouletteStartDepth < 1 || config.numeric.maxRootIterations < 1 ||
        config.numeric.maxQuadratureSubdivisions < 1 ||
        !(config.numeric.distanceAbsoluteTolerance > 0.0) || !(config.numeric.distanceRelativeTolerance > 0.0) ||
        !(config.numeric.relativeTolerance > 0.0) || !(config.numeric.absoluteTolerance > 0.0)) {
        throw std::invalid_argument("invalid experiment budget or numerical tolerance");
    }
    if (!std::isfinite(config.numeric.relativeTolerance) || !std::isfinite(config.numeric.absoluteTolerance) ||
        !std::isfinite(config.numeric.distanceAbsoluteTolerance) || !std::isfinite(config.numeric.distanceRelativeTolerance))
        throw std::invalid_argument("numerical tolerances must be finite");
    config.field.validate();
    config.material.validate(config.field.activeDomain);
    defaultNumericPolicy() = config.numeric;
    return config;
}

void writeResolvedConfig(const ExperimentConfig& config, const std::filesystem::path& path) {
    nlohmann::json result;
    result["schema_version"] = config.schemaVersion;
    result["seed"] = config.seed;
    if (const auto* cutaway = dynamic_cast<const CutawaySphereMean*>(config.field.mean.get())) {
        result["field"] = {{"mean_type", "cutaway_sphere"},
                           {"sphere_center", toArray(cutaway->center())},
                           {"inner_radius", cutaway->innerRadius()},
                           {"outer_radius", cutaway->outerRadius()},
                           {"removed_wedge", "local_x > 0 && local_y > 0 outside inner sphere"}};
    } else if (const auto* shaderBall = dynamic_cast<const ShaderBallMean*>(config.field.mean.get())) {
        result["field"] = {{"mean_type", "shader_ball"},
                           {"sphere_center", toArray(shaderBall->center())},
                           {"sphere_radius", shaderBall->radius()},
                           {"groove_axis", toArray(shaderBall->grooveAxis())},
                           {"groove_radius", shaderBall->grooveRadius()}};
    } else if (const auto written = writeMeanToJson(*config.field.mean)) {
        // Types the core library does not know (a baked NanoVDB grid, say)
        // describe themselves through the same registry that built them.
        result["field"] = *written;
    }
    result["derived"] = {
        {"sigma", config.field.kernel.sigma()},
        {"kernel_precision", {{config.field.kernel.precision()(0,0), config.field.kernel.precision()(0,1), config.field.kernel.precision()(0,2)},
                              {config.field.kernel.precision()(1,0), config.field.kernel.precision()(1,1), config.field.kernel.precision()(1,2)},
                              {config.field.kernel.precision()(2,0), config.field.kernel.precision()(2,1), config.field.kernel.precision()(2,2)}}},
        {"domain_min", toArray(config.field.activeDomain.minimum)},
        {"domain_max", toArray(config.field.activeDomain.maximum)}};
    const double fieldVariance = config.field.kernel.sigma() * config.field.kernel.sigma();
    const Vector3 gradientStddev =
        (fieldVariance * config.field.kernel.precision().diagonal()).cwiseMax(0.0).cwiseSqrt();
    result["derived"]["gradient_stddev_xyz"] = toArray(gradientStddev);
    if (config.preparedAreaMajorant)
        result["derived"]["projected_area_majorant"] = *config.preparedAreaMajorant;
    result["material"]["ndf_family"] = config.material.ndfFamily == NdfFamily::GGXBaseline
        ? "ggx" : config.material.ndfFamily == NdfFamily::BeckmannLimit
            ? "beckmann_limit" : "generalized_gaussian";
    result["material"]["gp_model"] = config.material.gpModel == GpModel::GlobalPointwise
        ? "global_pointwise" : "local_tangent";
    if (result.contains("field") && result["field"].value("mean_type", "") == "nanovdb")
        result["field"]["use_alpha_grid"] = static_cast<bool>(config.material.alphaField);
    result["material"]["ggx_alpha"] =
        {config.material.ggxAlpha.x(), config.material.ggxAlpha.y()};
    if (config.materialRoughness) {
        result["material"]["roughness"] = *config.materialRoughness;
        result["material"]["roughness_definition"] =
            "Beckmann alpha convention for the isotropic generalized_gaussian gradient; "
            "Sigma_G = roughness^2 / 2 I; correlation length = sqrt(2) * sigma / roughness";
        result["derived"]["isotropic_correlation_length"] =
            config.field.kernel.sigma() / (*config.materialRoughness / std::sqrt(2.0));
    }
    result["budgets"] = {{"render_width", config.render.width},
                          {"render_height", config.render.height},
                          {"render_spp", config.render.samplesPerPixel},
                          {"render_threads", config.render.threadCount}};
    result["classic_phase_proposal"] = config.classicPhaseProposal;
    result["transport"]["mode"] = config.transportMode;
    result["transport"]["classic"]["sampler"] = "dda_null_tracking";
    result["transport"]["global_conditional"] = {
        {"sampler", "two_segment_delta_tracking"},
        {"majorant", "analytic_birth_and_conditional_interval_bounds"},
        {"tracking_segments", 2},
        {"far_interval_thinning", true},
        {"birth_policy", "sampled_positive_exterior"}};
    if (config.transmittance) {
        const TransmittanceConfig& source = *config.transmittance;
        nlohmann::json experiment;
        experiment["modes"] = source.modes;
        experiment["curve"] = {
            {"bins", source.bins},
            {"spacing", source.spacing},
            {"distance_origin", source.distanceOrigin}};
        experiment["monte_carlo"] = {
            {"trials_per_ray", source.trialsPerRay},
            {"confidence_level", source.confidenceLevel},
            {"write_raw_samples", source.writeRawSamples}};
        experiment["conditional_birth"] = conditionalBirthJson(source.conditionalBirth);
        experiment["comparison"] = {
            {"local_transport_reference", source.localTransportReference},
            {"constant_exponential", {
                {"enabled", source.constantExponentialEnabled},
                {"fit", source.constantExponentialFit}}}};
        experiment["rays"] = nlohmann::json::array();
        for (const TransmittanceRayConfig& ray : source.rays) {
            nlohmann::json entry{
                {"id", ray.id},
                {"group", ray.group},
                {"origin", toArray(ray.origin)},
                {"direction", toArray(ray.direction)},
                {"max_distance", ray.maximumDistance}};
            if (ray.bins) entry["bins"] = *ray.bins;
            if (ray.conditionalBirth)
                entry["conditional_birth"] = conditionalBirthJson(*ray.conditionalBirth);
            experiment["rays"].push_back(std::move(entry));
        }
        result["transmittance"] = std::move(experiment);
    }
    result["numeric"] = {{"relative_tolerance", config.numeric.relativeTolerance},
        {"absolute_tolerance", config.numeric.absoluteTolerance},
        {"distance_absolute_tolerance", config.numeric.distanceAbsoluteTolerance},
        {"distance_relative_tolerance", config.numeric.distanceRelativeTolerance},
        {"max_root_iterations", config.numeric.maxRootIterations},
        {"max_quadrature_subdivisions", config.numeric.maxQuadratureSubdivisions}};
    result["rng_stream"] = "per_pixel_v1";
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write resolved config");
    stream << std::setw(2) << result << '\n';
}

} // namespace mf

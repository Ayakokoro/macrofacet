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

SquaredExponentialKernel roughnessKernel(double sigma, double roughness, NdfFamily family) {
    requirePositiveFinite(sigma, "sigma");
    requirePositiveFinite(roughness, "material roughness");
    if (family != NdfFamily::GeneralizedGaussian) {
        throw std::invalid_argument("material roughness requires the generalized_gaussian NDF family");
    }
    const double valueVariance = sigma * sigma;
    const double gradientVariance = roughness * roughness;
    const double correlationLength = sigma / roughness;
    if (!(valueVariance > 0.0) || !std::isfinite(valueVariance) ||
        !(gradientVariance > 0.0) || !std::isfinite(gradientVariance) ||
        !(correlationLength > 0.0) || !std::isfinite(correlationLength)) {
        throw std::invalid_argument(
            "roughness and sigma must yield finite positive variances and correlation length");
    }
    const double inverseLength = roughness / sigma;
    const double precision = inverseLength * inverseLength;
    const double actualGradientVariance = valueVariance * precision;
    if (!(precision > 0.0) || !std::isfinite(precision) ||
        !(actualGradientVariance > 0.0) || !std::isfinite(actualGradientVariance)) {
        throw std::invalid_argument(
            "roughness / sigma must yield finite positive kernel precision and gradient covariance");
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

std::string modelModeName(ModelMode mode) {
    if (mode == ModelMode::Classic) return "classic";
    return "conditional29";
}

void requireNanoVdbField(const ExperimentConfig& config) {
    if (!config.mediumDensity || !config.field.mean ||
        std::string(config.field.mean->typeName()) != "nanovdb" ||
        !(config.field.mean->voxelSizeHint() > 0.0)) {
        throw std::invalid_argument("experiment tracing requires a prepared NanoVDB field");
    }
}

void applyFieldOverrides(ExperimentConfig& config, std::optional<double> sigma,
                         std::optional<double> roughness, bool preserveSlope) {
    if (sigma) requirePositiveFinite(*sigma, "sigma");
    // The CLI is the other way a sigma can reach a baked field, so it obeys the
    // same rule as the config key: sigma is part of the data, not a knob.
    // Sweeping it would reinterpret the band (+-3 sigma) and the background
    // (+6 sigma) at the wrong scale and render a plausible but wrong surface.
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
    config.modes.clear();
    for (const auto& modeValue : root.at("modes")) {
        const std::string mode = modeValue.get<std::string>();
        if (mode == "classic") config.modes.push_back(ModelMode::Classic);
        else if (mode == "conditional29") config.modes.push_back(ModelMode::Conditional29);
        else throw std::invalid_argument("unknown model mode: " + mode);
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
    const auto alpha = material.contains("ggx_alpha") ? material.at("ggx_alpha") :
        fieldJson.value("ggx_alpha", nlohmann::json());
    if (!alpha.is_null()) {
        config.material.ggxAlpha = Vector2(alpha[0].get<double>(), alpha[1].get<double>());
    }
    config.material.alphaField = std::move(built.alphaField);

    config.material.conductor.eta = vector3(material.at("eta_rgb"));
    config.material.conductor.k = vector3(material.at("k_rgb"));
    config.material.conductor.forceUnitFresnel =
        material.value("force_unit_fresnel_for_energy_test", false);
    const auto& transport = root.at("transport");
    const auto conditional = transport.value("conditional29", nlohmann::json::object());
    if (!conditional.is_object()) throw std::invalid_argument("conditional29 transport must be an object");
    const std::string sampler = conditional.value("sampler", transport.value("correlated_sampler", "optical_depth"));
    if (sampler != "regular_tracking" && sampler != "optical_depth")
        throw std::invalid_argument("conditional29 supports only regular_tracking (optical_depth alias)");
    if (conditional.value("optical_depth_solver", "safeguarded_newton") != "safeguarded_newton")
        throw std::invalid_argument("conditional29 requires safeguarded_newton");
    const std::string commonExternal = transport.value("external_policy", "original_macrofacet");
    if (commonExternal != "original_macrofacet" && commonExternal != "sampled_exterior")
        throw std::invalid_argument("unknown external policy");
    const std::string external = conditional.value("external_policy", commonExternal);
    if (external == "original_macrofacet") config.conditional29.externalPolicy = ExternalPolicy::OriginalMacrofacet;
    else if (external == "sampled_exterior") config.conditional29.externalPolicy = ExternalPolicy::SampledExterior;
    else throw std::invalid_argument("unknown conditional29 external policy");
    if (transport.contains("hard_depth_cap") && !transport["hard_depth_cap"].is_null()) {
        config.conditional29.hardDepthCap = transport["hard_depth_cap"].get<int>();
        if (*config.conditional29.hardDepthCap < 1) throw std::invalid_argument("hard_depth_cap must be positive or null");
    }
    if (transport.value("next_event_estimation", false) &&
        std::find(config.modes.begin(), config.modes.end(), ModelMode::Conditional29) != config.modes.end())
        throw std::invalid_argument("conditional29 NEE requires an extended gradient-state light sampler and is not implemented");
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

    const auto& flight = root.at("fixed_flight");
    config.fixedFlight.birthPosition = vector3(flight.at("birth_position"));
    config.fixedFlight.birthGradient = vector3(flight.at("birth_gradient"));
    config.fixedFlight.direction = normalizedOrThrow(vector3(flight.at("direction")));
    config.fixedFlight.requestedMaximumAge = flight.at("requested_maximum_age").get<double>();
    config.fixedFlight.curveSampleCount = flight.at("curve_sample_count").get<int>();
    config.fixedFlight.flightSampleCount = flight.at("flight_sample_count").get<int>();

    const auto& reference = root.at("reference");
    config.reference.pathSampleCount = reference.at("path_sample_count").get<int>();
    config.reference.nestedGridIntervals = reference.at("nested_grid_intervals").get<std::vector<int>>();
    config.reference.formulaCheckpointCounts =
        reference.at("formula_checkpoint_counts").get<std::vector<int>>();
    config.reference.formulaAgeCount = reference.at("formula_age_count").get<int>();
    config.reference.formulaSampleCount = reference.at("formula_sample_count").get<int>();
    config.reference.maxGridPoints = reference.at("max_grid_points").get<int>();

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
    config.render.flightTableCells = render.value("flight_table_cells", 48);
    config.render.threadCount = render.value("thread_count", 0);
    config.outputDirectory = root.at("output_directory").get<std::string>();

    if (config.fixedFlight.direction.dot(config.fixedFlight.birthGradient) <= 0.0) {
        throw std::invalid_argument("fixed-flight direction must depart along the exterior gradient");
    }
    if (config.fixedFlight.curveSampleCount < 2 || config.fixedFlight.flightSampleCount < 1 ||
        config.render.width < 1 || config.render.height < 1 || config.render.samplesPerPixel < 1 ||
        config.render.flightTableCells < 4 || config.render.threadCount < 0 ||
        config.render.rouletteStartDepth < 1 || config.numeric.maxRootIterations < 1 ||
        config.numeric.maxQuadratureSubdivisions < 1 ||
        !(config.numeric.distanceAbsoluteTolerance > 0.0) || !(config.numeric.distanceRelativeTolerance > 0.0) ||
        !(config.numeric.relativeTolerance > 0.0) || !(config.numeric.absoluteTolerance > 0.0)) {
        throw std::invalid_argument("invalid experiment budget or numerical tolerance");
    }
    if (!std::isfinite(config.numeric.relativeTolerance) || !std::isfinite(config.numeric.absoluteTolerance) ||
        !std::isfinite(config.numeric.distanceAbsoluteTolerance) || !std::isfinite(config.numeric.distanceRelativeTolerance))
        throw std::invalid_argument("numerical tolerances must be finite");
    if (config.material.ndfFamily == NdfFamily::GGXBaseline) {
        for (ModelMode mode : config.modes) if (mode != ModelMode::Classic) {
            throw std::invalid_argument("GGX is supported only by classic mode");
        }
    }
    config.field.validate();
    config.material.validate(config.field.activeDomain);
    defaultNumericPolicy() = config.numeric;
    return config;
}

void writeResolvedConfig(const ExperimentConfig& config, const std::filesystem::path& path) {
    nlohmann::json result;
    result["schema_version"] = config.schemaVersion;
    result["seed"] = config.seed;
    for (ModelMode mode : config.modes) result["modes"].push_back(modelModeName(mode));
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
        {"domain_max", toArray(config.field.activeDomain.maximum)},
        {"fixed_direction", toArray(config.fixedFlight.direction)}};
    const double fieldVariance = config.field.kernel.sigma() * config.field.kernel.sigma();
    const Vector3 gradientStddev =
        (fieldVariance * config.field.kernel.precision().diagonal()).cwiseMax(0.0).cwiseSqrt();
    result["derived"]["gradient_stddev_xyz"] = toArray(gradientStddev);
    result["material"]["ndf_family"] = config.material.ndfFamily == NdfFamily::GGXBaseline
        ? "ggx" : config.material.ndfFamily == NdfFamily::BeckmannLimit
            ? "beckmann_limit" : "generalized_gaussian";
    result["material"]["ggx_alpha"] =
        {config.material.ggxAlpha.x(), config.material.ggxAlpha.y()};
    if (config.materialRoughness) {
        result["material"]["roughness"] = *config.materialRoughness;
        result["material"]["roughness_definition"] =
            "standard deviation of each isotropic generalized_gaussian gradient component; "
            "Sigma_G = roughness^2 I; correlation length = sigma / roughness";
        result["derived"]["isotropic_correlation_length"] =
            config.field.kernel.sigma() / *config.materialRoughness;
    }
    result["budgets"] = {{"flight_samples", config.fixedFlight.flightSampleCount},
                          {"reference_path_samples", config.reference.pathSampleCount},
                          {"formula_samples", config.reference.formulaSampleCount},
                          {"render_width", config.render.width},
                          {"render_height", config.render.height},
                          {"render_spp", config.render.samplesPerPixel},
                          {"flight_table_cells", config.render.flightTableCells},
                          {"render_threads", config.render.threadCount}};
    result["classic_phase_proposal"] = config.classicPhaseProposal;
    result["transport"]["conditional29"] = {
        {"sampler", "regular_tracking"}, {"optical_depth_solver", "safeguarded_newton"},
        {"external_policy", config.conditional29.externalPolicy == ExternalPolicy::SampledExterior
            ? "sampled_exterior" : "original_macrofacet"},
        {"hard_depth_cap", config.conditional29.hardDepthCap
            ? nlohmann::json(*config.conditional29.hardDepthCap) : nlohmann::json(nullptr)},
        {"initial_integration_cells", config.render.flightTableCells}};
    result["transport"]["classic"]["sampler"] = config.mediumSurfaceBand
        ? "dda_null_tracking" : "regular_tracking";
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

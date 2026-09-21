#include "macrofacet/experiments/ExperimentConfig.h"
#include <nlohmann/json.hpp>
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

} // namespace

GPSSField buildDefaultField() {
    GPSSField field{
        std::make_shared<PlaneMean>(Vector3::UnitZ(), 0.0),
        SquaredExponentialKernel::fromCorrelationLengths(0.1, Vector3(0.2, 0.2, 0.2)),
        {Point3(-2.0, -2.0, -0.3), Point3(2.0, 2.0, 0.3)},
        {}, NdfFamily::GeneralizedGaussian, Vector2(0.5, 0.5)};
    field.validate();
    return field;
}

std::string modelModeName(ModelMode mode) {
    if (mode == ModelMode::Classic) return "classic";
    if (mode == ModelMode::Conditional29) return "conditional29";
    return "midpoint";
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
        else if (mode == "midpoint") config.modes.push_back(ModelMode::Midpoint);
        else throw std::invalid_argument("unknown model mode: " + mode);
    }

    const auto& fieldJson = root.at("field");
    const std::string meanType = fieldJson.at("mean_type").get<std::string>();
    MeanFieldPtr mean;
    if (meanType == "plane") {
        mean = std::make_shared<PlaneMean>(vector3(fieldJson.at("plane_normal")),
                                           fieldJson.at("plane_offset").get<double>());
    } else if (meanType == "sphere") {
        mean = std::make_shared<SphereMean>(vector3(fieldJson.at("sphere_center")),
                                            fieldJson.at("sphere_radius").get<double>());
    } else if (meanType == "constant") {
        mean = std::make_shared<ConstantMean>(fieldJson.at("constant_value").get<double>());
    } else {
        throw std::invalid_argument("unknown mean type: " + meanType);
    }
    const double sigma = fieldJson.at("sigma").get<double>();
    Vector3 lengths;
    const auto& lengthJson = fieldJson.at("correlation_lengths");
    for (int i = 0; i < 3; ++i) {
        lengths[i] = lengthJson[i].is_null() ? std::numeric_limits<double>::infinity()
                                             : lengthJson[i].get<double>();
    }
    const Matrix3 rotation = matrix3(fieldJson.at("kernel_rotation"));
    const std::string family = fieldJson.at("ndf_family").get<std::string>();
    NdfFamily ndfFamily;
    if (family == "generalized_gaussian") ndfFamily = NdfFamily::GeneralizedGaussian;
    else if (family == "beckmann_limit") ndfFamily = NdfFamily::BeckmannLimit;
    else if (family == "ggx") ndfFamily = NdfFamily::GGXBaseline;
    else throw std::invalid_argument("unknown NDF family: " + family);
    config.field = {mean, SquaredExponentialKernel::fromCorrelationLengths(sigma, lengths, rotation),
                    {vector3(fieldJson.at("domain_min")), vector3(fieldJson.at("domain_max"))},
                    {}, ndfFamily, Vector2(0.5, 0.5)};
    if (fieldJson.contains("ggx_alpha")) {
        config.field.ggxAlpha = Vector2(fieldJson["ggx_alpha"][0].get<double>(),
                                        fieldJson["ggx_alpha"][1].get<double>());
    }

    const auto& material = root.at("material");
    config.field.conductor.eta = vector3(material.at("eta_rgb"));
    config.field.conductor.k = vector3(material.at("k_rgb"));
    config.field.conductor.forceUnitFresnel =
        material.value("force_unit_fresnel_for_energy_test", false);
    const auto& transport = root.at("transport");
    if (transport.at("external_policy").get<std::string>() != "original_macrofacet") {
        throw std::invalid_argument("only original_macrofacet external policy is supported");
    }
    config.beckmannMixtureWeight = transport.value("beckmann_mixture_weight", 0.5);
    config.classicPhaseProposal = transport.value("classic_phase_proposal", "uniform");
    if (config.classicPhaseProposal != "uniform" &&
        config.classicPhaseProposal != "paper_mixture" &&
        config.classicPhaseProposal != "target_vndf") {
        throw std::invalid_argument("unknown classic phase proposal");
    }
    if (config.classicPhaseProposal == "target_vndf" &&
        config.field.ndfFamily == NdfFamily::GGXBaseline) {
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
    config.reference.confidenceLevel = reference.at("confidence_level").get<double>();
    config.reference.maxGridPoints = reference.at("max_grid_points").get<int>();

    const auto& numeric = root.at("numeric");
    config.numeric.relativeTolerance = numeric.at("relative_tolerance").get<double>();
    config.numeric.absoluteTolerance = numeric.at("absolute_tolerance").get<double>();
    config.numeric.maxQuadratureSubdivisions = numeric.at("max_quadrature_subdivisions").get<int>();
    config.numeric.maxRootIterations = numeric.at("max_root_iterations").get<int>();
    config.numeric.allowHigherPrecisionFallback = numeric.value("higher_precision_fallback", true);
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
    if (!config.field.activeDomain.contains(config.fixedFlight.birthPosition, 1e-12)) {
        throw std::invalid_argument("fixed-flight birth lies outside the active domain");
    }
    if (config.fixedFlight.curveSampleCount < 2 || config.fixedFlight.flightSampleCount < 1 ||
        config.render.width < 1 || config.render.height < 1 || config.render.samplesPerPixel < 1 ||
        config.render.flightTableCells < 4 || config.render.threadCount < 0 ||
        !(config.numeric.relativeTolerance > 0.0) || !(config.numeric.absoluteTolerance > 0.0)) {
        throw std::invalid_argument("invalid experiment budget or numerical tolerance");
    }
    if (config.field.ndfFamily == NdfFamily::GGXBaseline) {
        for (ModelMode mode : config.modes) if (mode != ModelMode::Classic) {
            throw std::invalid_argument("GGX is supported only by classic mode");
        }
    }
    config.field.validate();
    defaultNumericPolicy() = config.numeric;
    return config;
}

void writeResolvedConfig(const ExperimentConfig& config, const std::filesystem::path& path) {
    nlohmann::json result;
    result["schema_version"] = config.schemaVersion;
    result["seed"] = config.seed;
    for (ModelMode mode : config.modes) result["modes"].push_back(modelModeName(mode));
    result["derived"] = {
        {"sigma", config.field.kernel.sigma()},
        {"kernel_precision", {{config.field.kernel.precision()(0,0), config.field.kernel.precision()(0,1), config.field.kernel.precision()(0,2)},
                              {config.field.kernel.precision()(1,0), config.field.kernel.precision()(1,1), config.field.kernel.precision()(1,2)},
                              {config.field.kernel.precision()(2,0), config.field.kernel.precision()(2,1), config.field.kernel.precision()(2,2)}}},
        {"domain_min", toArray(config.field.activeDomain.minimum)},
        {"domain_max", toArray(config.field.activeDomain.maximum)},
        {"fixed_direction", toArray(config.fixedFlight.direction)}};
    result["budgets"] = {{"flight_samples", config.fixedFlight.flightSampleCount},
                          {"reference_path_samples", config.reference.pathSampleCount},
                          {"formula_samples", config.reference.formulaSampleCount},
                          {"render_width", config.render.width},
                          {"render_height", config.render.height},
                          {"render_spp", config.render.samplesPerPixel},
                          {"flight_table_cells", config.render.flightTableCells},
                          {"render_threads", config.render.threadCount}};
    result["classic_phase_proposal"] = config.classicPhaseProposal;
    result["rng_stream"] = "per_pixel_v1";
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write resolved config");
    stream << std::setw(2) << result << '\n';
}

} // namespace mf

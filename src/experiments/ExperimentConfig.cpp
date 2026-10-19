#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/learned/CausalFirstHitModel.h"
#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/gpss/ShaderBallMean.h"
#include "macrofacet/transport/RenewalMedium.h"
#include "macrofacet/transport/FirstHitSEMedium.h"
#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
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
           mode == "global_conditional" || mode == "global_conditional_renewal_plus";
}

nlohmann::json conditionalBirthJson(const ConditionalBirthConfig& birth) {
    nlohmann::json result{{"policy", birth.policy}};
    if (birth.policy == "fixed_observation") {
        result["value"] = birth.value;
        result["gradient"] = toArray(birth.gradient);
    }
    return result;
}

CovarianceKernel roughnessKernel(CovarianceKernelType type, double sigma,
                                 double roughness, NdfFamily family) {
    requirePositiveFinite(sigma, "sigma");
    requirePositiveFinite(roughness, "material roughness");
    if (family != NdfFamily::GeneralizedGaussian) {
        throw std::invalid_argument("material roughness requires the generalized_gaussian NDF family");
    }
    // Beckmann alpha is sqrt(2) times the standard deviation of each slope
    // component when the mean height gradient has unit length.
    const double gradientStddev = roughness / std::sqrt(2.0);
    const double gradientVariance = gradientStddev * gradientStddev;
    if (!(gradientVariance > 0.0) || !std::isfinite(gradientVariance)) {
        throw std::invalid_argument(
            "roughness and sigma must yield finite positive variances and correlation length");
    }
    return CovarianceKernel::fromGradientCovariance(
        type, sigma, gradientVariance * Matrix3::Identity());
}

TungstenFunctionSpaceConfig readTungstenSampler(const nlohmann::json& object) {
    const std::unordered_set<std::string> allowed{"sample_points","step_size","correlation_context",
        "intersect_method","normal_method","skip_space"};
    for (const auto& item : object.items())
        if (!allowed.count(item.key())) throw std::invalid_argument("unsupported Tungsten sampler option: "+item.key());
    if (object.value("correlation_context","renewal+")!="renewal+" ||
        object.value("intersect_method","gp_discrete")!="gp_discrete" ||
        object.value("normal_method","conditioned_gaussian")!="conditioned_gaussian" ||
        object.value("skip_space",0.)!=0.)
        throw std::invalid_argument("Tungsten compatibility requires renewal+, gp_discrete, conditioned_gaussian, skip_space=0");
    TungstenFunctionSpaceConfig result;
    result.samplePoints=object.value("sample_points",8);
    result.stepSize=object.value("step_size",0.01);
    result.validate();
    return result;
}

nlohmann::json tungstenSamplerJson(const TungstenFunctionSpaceConfig& config) {
    return {{"sample_points",config.samplePoints},{"step_size",config.stepSize},
        {"correlation_context","renewal+"},{"intersect_method","gp_discrete"},
        {"normal_method","conditioned_gaussian"},{"skip_space",0.0}};
}

} // namespace

GPSSField buildDefaultField() {
    GPSSField field{
        std::make_shared<PlaneMean>(Vector3::UnitZ(), 0.0),
        CovarianceKernel::fromCorrelationLengths(
            CovarianceKernelType::SquaredExponential, 0.1, Vector3(0.2, 0.2, 0.2)),
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

void prepareRenewalModel(ExperimentConfig& config) {
    if (config.transportMode == "global_conditional_renewal_plus") {
        if (config.material.ndfFamily != NdfFamily::GeneralizedGaussian || config.material.alphaField)
            throw std::invalid_argument("global_conditional_renewal_plus requires generalized_gaussian without alpha grid");
        (void)GlobalConditionalRenewalPlusMedium(config.field,config.conditionalRenewalDistance);
        if (config.material.gpModel != GpModel::GlobalPointwise) config.preparedAreaMajorant.reset();
        config.material.gpModel = GpModel::GlobalPointwise;
        return;
    }
    if (config.transportMode != "neural_renewal" && config.transportMode != "hybrid_first_hit") return;
    if (config.transportMode == "hybrid_first_hit") {
        if ((config.continuationMethod != "neural_renewal" && config.continuationMethod != "classic_local" &&
             config.continuationMethod != "global_conditional" && config.continuationMethod != "global_conditional_renewal_plus" &&
             config.continuationMethod != kTungstenFunctionSpaceMethod) ||
            config.renewal.backend != "scalar")
            throw std::invalid_argument("unsupported hybrid continuation method or non-scalar backend");
        if (config.continuationMethod != "neural_renewal") {
            const bool tungsten = config.continuationMethod == kTungstenFunctionSpaceMethod;
            const bool conditional = config.continuationMethod == "global_conditional" ||
                config.continuationMethod == "global_conditional_renewal_plus" || tungsten;
            if (conditional) {
                if (config.material.ndfFamily != NdfFamily::GeneralizedGaussian || config.material.alphaField)
                    throw std::invalid_argument("global_conditional requires generalized_gaussian without alpha grid");
                if (!config.field.kernel.supportsAnalyticConditionalTransport())
                    throw std::invalid_argument("global_conditional requires a squared_exponential kernel");
            }
            if (tungsten) (void)TungstenFunctionSpaceSampler(config.field,config.continuationTungsten);
            if (config.continuationMethod == "global_conditional_renewal_plus")
                (void)GlobalConditionalRenewalPlusMedium(config.field,config.conditionalRenewalDistance);
            const auto gpModel = conditional ? GpModel::GlobalPointwise : GpModel::LocalTangent;
            if (config.material.gpModel != gpModel) config.preparedAreaMajorant.reset();
            config.material.gpModel = gpModel;
            config.renewal.resolvedBackend = "scalar";
            return; // Analytic continuation does not load a renewal model.
        }
    }
    if (config.renewal.batchSize < 1 || config.renewal.batchSize > 65536)
        throw std::invalid_argument("transport.renewal.batch_size must be 1..65536");
    if (config.renewal.rayPoolSize < 0 || config.renewal.rayPoolSize > 65536)
        throw std::invalid_argument("transport.renewal.ray_pool_size must be 0..65536");
    if (config.renewal.auxiliaryBatchMinimum < 1 || config.renewal.auxiliaryBatchMinimum > 65536)
        throw std::invalid_argument("transport.renewal.auxiliary_batch_minimum must be 1..65536");
    if (config.renewal.maximumQueueDelay < 0 || config.renewal.maximumQueueDelay > 65536)
        throw std::invalid_argument("transport.renewal.maximum_queue_delay must be 0..65536");
    if (config.renewal.maximumInFlightBatches < 1 || config.renewal.maximumInFlightBatches > 4)
        throw std::invalid_argument("transport.renewal.max_in_flight_batches must be 1..4");
    config.renewal.resolvedBackend = resolveRenewalBackend(config.renewal.backend);
    if (config.material.ndfFamily != NdfFamily::GeneralizedGaussian || config.material.alphaField)
        throw std::invalid_argument("neural_renewal requires generalized_gaussian and field.use_alpha_grid=false");
    if (!config.renewal.model) {
        if (config.renewal.modelPath.empty())
            throw std::invalid_argument("neural_renewal requires transport.renewal.model");
        config.renewal.model = std::make_shared<const RenewalHazardModel>(
            RenewalHazardModel::load(config.renewal.modelPath));
    }
    (void)RenewalMedium(config.field, *config.renewal.model, config.renewal.profileMaximumStep,
                        config.renewal.profileMode);
    if (config.material.gpModel != GpModel::GlobalPointwise) config.preparedAreaMajorant.reset();
    config.material.gpModel = GpModel::GlobalPointwise;
}

void prepareFirstHitModel(ExperimentConfig& config) {
    if (config.transportMode != "hybrid_first_hit") return;
    if (config.firstHit.method == "classic_local") {
        if (config.firstHit.backend != "scalar" || config.material.ndfFamily != NdfFamily::GeneralizedGaussian ||
            config.material.alphaField)
            throw std::invalid_argument("classic_local first-hit requires scalar generalized_gaussian without alpha grid");
        config.firstHit.model.reset();
        config.firstHit.profileMode = "local_direct";
        return;
    }
    if (config.firstHit.method == "global_conditional" ||
        config.firstHit.method == "global_conditional_renewal_plus") {
        if (config.firstHit.backend != "scalar" || config.material.ndfFamily != NdfFamily::GeneralizedGaussian ||
            config.material.alphaField)
            throw std::invalid_argument(config.firstHit.method + " first-hit requires scalar generalized_gaussian without alpha grid");
        if (!config.field.kernel.supportsAnalyticConditionalTransport())
            throw std::invalid_argument(config.firstHit.method + " first-hit requires a squared_exponential kernel");
        if (config.firstHit.method == "global_conditional_renewal_plus")
            (void)GlobalConditionalRenewalPlusMedium(config.field,config.firstHit.renewalDistance);
        config.firstHit.model.reset();
        config.firstHit.profileMode = "conditional_direct";
        return;
    }
    if (config.firstHit.method == kTungstenFunctionSpaceMethod) {
        if (config.firstHit.backend!="scalar" || config.material.ndfFamily!=NdfFamily::GeneralizedGaussian ||
            config.material.alphaField) throw std::invalid_argument("Tungsten first-hit requires scalar generalized_gaussian without alpha grid");
        config.firstHit.model.reset();
        config.firstHit.profileMode="tungsten_direct";
        (void)TungstenFunctionSpaceSampler(config.field,config.firstHit.tungsten);
        return;
    }
    if (config.firstHit.method != "neural_first_hit_se" && config.firstHit.method != "gpis_function_space")
        throw std::invalid_argument("unknown transport.first_hit.method");
    if (config.firstHit.profileMode != "cubic")
        throw std::invalid_argument("first-hit rendering requires cubic profiles");
    if (config.material.ndfFamily != NdfFamily::GeneralizedGaussian || config.material.alphaField)
        throw std::invalid_argument("first-hit rendering requires generalized_gaussian without alpha grid");
    if (config.firstHit.method == "neural_first_hit_se" && !config.firstHit.model) {
        if (config.firstHit.modelPath.empty()) throw std::invalid_argument("transport.first_hit.model is required");
        config.firstHit.model = std::make_shared<const FirstHitSEModel>(FirstHitSEModel::load(config.firstHit.modelPath));
    }
    if (config.firstHit.method == "gpis_function_space") config.firstHit.model.reset();
    if (config.firstHit.batchSize<1 || config.firstHit.batchSize>65536)
        throw std::invalid_argument("first-hit batch_size must be in [1,65536]");
    if (config.firstHit.backend!="scalar") {
        if (!config.firstHit.model || !config.firstHit.model->causal())
            throw std::invalid_argument("batched first-hit rendering requires a causal residual GRU model");
        CausalFirstHitSession::validateBackend(config.firstHit.backend);
    }
    (void)FirstHitSEMedium(config.field,config.firstHit.model,
        config.firstHit.profileMaximumStep,config.firstHit.reference);
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
        config.field.kernel = roughnessKernel(
            config.field.kernel.type(), sigma.value_or(config.field.kernel.sigma()),
            *effectiveRoughness, config.material.ndfFamily);
        config.materialRoughness = effectiveRoughness;
        config.field.validate();
    } else if (sigma) {
        const CovarianceKernelType type = config.field.kernel.type();
        const Matrix3 metric = config.field.kernel.metric();
        if (preserveSlope) {
            config.field.kernel = CovarianceKernel::fromGradientCovariance(
                type, *sigma, config.field.kernel.gradientCovarianceAtZero());
        } else {
            config.field.kernel = CovarianceKernel(type, *sigma, metric);
        }
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
    const CovarianceKernelType kernelType = parseCovarianceKernelType(
        fieldJson.value("kernel_type", std::string("squared_exponential")));
    CovarianceKernel kernel;
    if (material.contains("roughness")) {
        if (fieldJson.contains("correlation_lengths")) {
            throw std::invalid_argument("material.roughness and field.correlation_lengths are mutually exclusive");
        }
        if (!material.at("roughness").is_number()) {
            throw std::invalid_argument("material.roughness must be a finite positive number");
        }
        config.materialRoughness = material.at("roughness").get<double>();
        kernel = roughnessKernel(kernelType, sigma, *config.materialRoughness, ndfFamily);
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
        kernel = CovarianceKernel::fromCorrelationLengths(
            kernelType, sigma, lengths, rotation);
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
        config.transportMode != "global_conditional" && config.transportMode != "global_conditional_renewal_plus" &&
        config.transportMode != "neural_renewal" &&
        config.transportMode != "hybrid_first_hit" && config.transportMode != "all")
        throw std::invalid_argument("unknown transport.mode: " + config.transportMode);
    if (transport.contains("local_conditional"))
        throw std::invalid_argument("transport.local_conditional was removed; use transport.global_conditional_renewal_plus.renewal_distance");
    if (transport.contains("global_conditional_renewal_plus")) {
        const auto& renewal = transport.at("global_conditional_renewal_plus");
        if (renewal.contains("step_size"))
            throw std::invalid_argument("global_conditional_renewal_plus uses renewal_distance, not step_size");
        config.conditionalRenewalDistance = renewal.value("renewal_distance",0.01);
    }
    requirePositiveFinite(config.conditionalRenewalDistance,"transport.global_conditional_renewal_plus.renewal_distance");
    const nlohmann::json* renewalJson = transport.contains("renewal") ? &transport.at("renewal") : nullptr;
    if (config.transportMode == "hybrid_first_hit") {
        if (renewalJson) throw std::invalid_argument("hybrid_first_hit uses transport.continuation, not transport.renewal");
        const auto& first = transport.at("first_hit");
        config.firstHit.method = first.at("method").get<std::string>();
        config.firstHit.modelPath = first.value("model", "");
        const bool tungstenFirst = config.firstHit.method == kTungstenFunctionSpaceMethod;
        const bool conditionalRenewalFirst = config.firstHit.method == "global_conditional_renewal_plus";
        const bool conditionalFirst = config.firstHit.method == "global_conditional" || conditionalRenewalFirst;
        const bool localFirst = config.firstHit.method == "classic_local";
        const std::string profileMode = tungstenFirst ? "tungsten_direct" : conditionalFirst ? "conditional_direct" :
            localFirst ? "local_direct" : "cubic";
        config.firstHit.profileMode = first.value("profile_mode", profileMode);
        config.firstHit.profileMaximumStep = first.value("profile_maximum_step", 0.25);
        config.firstHit.backend = first.value("backend", "scalar");
        config.firstHit.batchSize = first.value("batch_size",1024);
        if (config.firstHit.method != "neural_first_hit_se" && config.firstHit.method != "gpis_function_space" &&
            !tungstenFirst && !conditionalFirst && !localFirst)
            throw std::invalid_argument("unknown transport.first_hit.method");
        if (config.firstHit.profileMode != profileMode)
            throw std::invalid_argument("first-hit profile_mode must match its sampler");
        if (config.firstHit.backend!="scalar" && (config.firstHit.method!="neural_first_hit_se" ||
            (config.firstHit.backend!="torch_cpu" && config.firstHit.backend!="torch_cuda")))
            throw std::invalid_argument("unsupported first-hit backend");
        if (config.firstHit.batchSize<1 || config.firstHit.batchSize>65536)
            throw std::invalid_argument("first-hit batch_size must be in [1,65536]");
        requirePositiveFinite(config.firstHit.profileMaximumStep,"transport.first_hit.profile_maximum_step");
        if (conditionalRenewalFirst) {
            if (first.contains("step_size"))
                throw std::invalid_argument("global_conditional_renewal_plus first-hit uses renewal_distance, not step_size");
            config.firstHit.renewalDistance = first.value("renewal_distance",0.01);
            requirePositiveFinite(config.firstHit.renewalDistance,"transport.first_hit.renewal_distance");
        } else if (first.contains("renewal_distance")) {
            throw std::invalid_argument("transport.first_hit.renewal_distance requires global_conditional_renewal_plus");
        }
        if (conditionalFirst || localFirst) {
            if (first.contains("model") || first.contains("profile_maximum_step") || first.contains("sampler") ||
                first.contains("batch_size"))
                throw std::invalid_argument(config.firstHit.method +
                    " first-hit uses direct delta tracking, not a model or profile sampler");
        } else if (tungstenFirst) {
            if (first.contains("model") || first.contains("profile_maximum_step"))
                throw std::invalid_argument("Tungsten first-hit uses direct field queries, not a model or profile step");
            config.firstHit.tungsten=readTungstenSampler(first.value("sampler",nlohmann::json::object()));
        } else if (first.contains("sampler")) {
            const auto& sampler = first.at("sampler");
            auto& r = config.firstHit.reference;
            r.historyPoints = sampler.value("history_points",1);
            r.batchSize = sampler.value("batch_size",8);
            r.step = sampler.value("step_size",1.0/64.0);
            r.rankTolerance = sampler.value("rank_tolerance",1e-10);
            r.crossingRule = sampler.value("crossing_rule","tungsten_linear");
        }
        (void)FunctionSpaceRaySampler(config.firstHit.reference);
        const auto& continuation = transport.at("continuation");
        config.continuationMethod = continuation.at("method").get<std::string>();
        config.renewal.backend = continuation.value("backend","scalar");
        if ((config.continuationMethod != "neural_renewal" && config.continuationMethod != "classic_local" &&
             config.continuationMethod != "global_conditional" && config.continuationMethod != "global_conditional_renewal_plus" &&
             config.continuationMethod != kTungstenFunctionSpaceMethod) ||
            config.renewal.backend != "scalar")
            throw std::invalid_argument("unsupported hybrid continuation method or non-scalar backend");
        if (config.continuationMethod == "global_conditional_renewal_plus") {
            if (continuation.contains("step_size"))
                throw std::invalid_argument("global_conditional_renewal_plus continuation uses renewal_distance, not step_size");
            config.conditionalRenewalDistance = continuation.value("renewal_distance",config.conditionalRenewalDistance);
            requirePositiveFinite(config.conditionalRenewalDistance,"transport.continuation.renewal_distance");
        }
        if (config.continuationMethod == kTungstenFunctionSpaceMethod)
            config.continuationTungsten=readTungstenSampler(continuation.value("sampler",nlohmann::json::object()));
        renewalJson = config.continuationMethod == "neural_renewal" ? &continuation : nullptr;
    }
    if (renewalJson) {
        const auto& renewal = *renewalJson;
        config.renewal.modelPath = renewal.at("model").get<std::string>();
        config.renewal.profileMaximumStep = renewal.value("profile_maximum_step", 0.25);
        config.renewal.profileMode = renewal.value("profile_mode", "cubic");
        if (config.renewal.profileMode != "cubic" && config.renewal.profileMode != "point_linear")
            throw std::invalid_argument("transport.renewal.profile_mode must be cubic or point_linear");
        requirePositiveFinite(config.renewal.profileMaximumStep, "transport.renewal.profile_maximum_step");
        config.renewal.backend = renewal.value("backend", "scalar");
        config.renewal.batchSize = renewal.value("batch_size", 4096);
        config.renewal.rayPoolSize = renewal.value("ray_pool_size", 0);
        config.renewal.auxiliaryBatchMinimum = renewal.value("auxiliary_batch_minimum", 256);
        config.renewal.maximumQueueDelay = renewal.value("maximum_queue_delay", 4);
        config.renewal.maximumInFlightBatches = renewal.value("max_in_flight_batches", 2);
        if (config.renewal.backend != "scalar" && config.renewal.backend != "torch_cpu" &&
            config.renewal.backend != "torch_cuda" && config.renewal.backend != "auto")
            throw std::invalid_argument("unknown transport.renewal.backend");
        if (config.renewal.batchSize < 1 || config.renewal.batchSize > 65536)
            throw std::invalid_argument("transport.renewal.batch_size must be 1..65536");

        if (config.renewal.rayPoolSize < 0 || config.renewal.rayPoolSize > 65536)
            throw std::invalid_argument("transport.renewal.ray_pool_size must be 0..65536");
        if (config.renewal.auxiliaryBatchMinimum < 1 || config.renewal.auxiliaryBatchMinimum > 65536)
            throw std::invalid_argument("transport.renewal.auxiliary_batch_minimum must be 1..65536");
        if (config.renewal.maximumQueueDelay < 0 || config.renewal.maximumQueueDelay > 65536)
            throw std::invalid_argument("transport.renewal.maximum_queue_delay must be 0..65536");
        if (config.renewal.maximumInFlightBatches < 1 || config.renewal.maximumInFlightBatches > 4)
            throw std::invalid_argument("transport.renewal.max_in_flight_batches must be 1..4");
    }
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
    if (transport.contains("first_passage_model")) {
        throw std::invalid_argument("transport.first_passage_model was removed; the legacy learned model is no longer supported");
    }

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
    config.render.writeFirstHitSamples = render.value("write_first_hit_samples", false);
    config.render.safetyDepthCap = render.value("safety_depth_cap", config.render.safetyDepthCap);

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
        config.render.threadCount < 0 || config.render.safetyDepthCap < 1 ||
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
    if (result.contains("field")) {
        result["field"]["kernel_type"] =
            covarianceKernelTypeName(config.field.kernel.type());
    }
    result["derived"] = {
        {"sigma", config.field.kernel.sigma()},
        {"kernel_type", covarianceKernelTypeName(config.field.kernel.type())},
        {"kernel_metric", {{config.field.kernel.metric()(0,0), config.field.kernel.metric()(0,1), config.field.kernel.metric()(0,2)},
                           {config.field.kernel.metric()(1,0), config.field.kernel.metric()(1,1), config.field.kernel.metric()(1,2)},
                           {config.field.kernel.metric()(2,0), config.field.kernel.metric()(2,1), config.field.kernel.metric()(2,2)}}},
        {"domain_min", toArray(config.field.activeDomain.minimum)},
        {"domain_max", toArray(config.field.activeDomain.maximum)}};
    // Retain the historical key for downstream readers. For non-SE kernels it
    // is the shared spatial metric, not the gradient covariance by itself.
    result["derived"]["kernel_precision"] = result["derived"]["kernel_metric"];
    if (config.field.kernel.type()==CovarianceKernelType::Matern32)
        result["derived"]["kernel_parameterization"]="unit_decay";
    if (config.field.kernel.type()==CovarianceKernelType::SquaredExponential)
        result["derived"]["kernel_parameterization"]="unit_length";
    const Vector3 gradientStddev =
        config.field.kernel.gradientCovarianceAtZero().diagonal().cwiseMax(0.0).cwiseSqrt();
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
            "Sigma_G = roughness^2 / 2 I; correlation length is kernel-family dependent";
        result["derived"]["isotropic_correlation_length"] =
            1.0 / std::sqrt(config.field.kernel.metric()(0, 0));
    }
    result["budgets"] = {{"render_width", config.render.width},
                          {"render_height", config.render.height},
                          {"render_spp", config.render.samplesPerPixel},
                          {"render_threads", config.render.threadCount}};
    result["classic_phase_proposal"] = config.classicPhaseProposal;
    result["transport"]["mode"] = config.transportMode;
    result["transport"]["global_conditional_renewal_plus"] = {
        {"renewal_distance",config.conditionalRenewalDistance},{"distance_units","world"},
        {"sampler","segmented_delta_tracking"},{"gp_model","global_pointwise"},
        {"conditioning","current_segment_value_and_full_gradient"},
        {"boundary_policy","sample_previous_segment_positive_endpoint"},
        {"mean_policy","global_field"},{"kernel_policy","shared_global_kernel"},
        {"coordinate_system","world"},
        {"majorant","analytic_birth_and_conditional_interval_bounds"}};
    if ((!config.renewal.modelPath.empty() || config.renewal.model) &&
        !(config.transportMode == "hybrid_first_hit" && config.continuationMethod != "neural_renewal")) {
        result["transport"]["renewal"] = {
            {"model", config.renewal.modelPath.string()},
            {"profile_maximum_step", config.renewal.profileMaximumStep},
            {"profile_mode", config.renewal.profileMode},
            {"backend", config.renewal.backend}, {"resolved_backend", config.renewal.resolvedBackend},
            {"batch_size", config.renewal.batchSize},
            {"ray_pool_size", config.renewal.rayPoolSize},
            {"auxiliary_batch_minimum", config.renewal.auxiliaryBatchMinimum},
            {"maximum_queue_delay", config.renewal.maximumQueueDelay},
            {"max_in_flight_batches", config.renewal.maximumInFlightBatches},
            {"sampler", "cumulative_inversion"}, {"normal", "full_renewal_plus"}};
        if (config.renewal.model) {
            result["transport"]["renewal"]["checkpoint_sha256"] = config.renewal.model->checkpointSha256();
            result["transport"]["renewal"]["model_kernel_type"] = covarianceKernelTypeName(config.renewal.model->kernelType());
        }
    }
    if (config.transportMode == "hybrid_first_hit") {
        if (config.continuationMethod == "classic_local") {
            result["transport"]["continuation"] = {
                {"backend","scalar"},{"resolved_backend","scalar"},
                {"sampler","dda_null_tracking"},{"gp_model","local_tangent"},
                {"classic_phase_proposal",config.classicPhaseProposal}};
        } else if (config.continuationMethod == "global_conditional_renewal_plus") {
            result["transport"]["continuation"] = result["transport"]["global_conditional_renewal_plus"];
            result["transport"]["continuation"]["backend"] = "scalar";
            result["transport"]["continuation"]["resolved_backend"] = "scalar";
            result["transport"]["continuation"]["birth_policy"] = "first_hit_surface";
        } else if (config.continuationMethod == "global_conditional") {
            result["transport"]["continuation"] = {
                {"backend","scalar"},{"resolved_backend","scalar"},
                {"sampler","segmented_delta_tracking"},{"gp_model","global_pointwise"},
                {"majorant","analytic_birth_and_conditional_interval_bounds"},
                {"birth_policy","first_hit_surface"},
                {"conditioning","last_collision_value_and_full_gradient"}};
        } else if (config.continuationMethod == kTungstenFunctionSpaceMethod) {
            result["transport"]["continuation"]={{"backend","scalar"},
                {"sampler",tungstenSamplerJson(config.continuationTungsten)},{"step_units","world"},
                {"conditioning","last_value_and_gradient_projected_on_new_direction"}};
        } else result["transport"]["continuation"] = result["transport"].at("renewal");
        result["transport"].erase("renewal");
        result["transport"]["continuation"]["method"] = config.continuationMethod;
        const auto& r = config.firstHit.reference;
        result["transport"]["first_hit"] = {
            {"method",config.firstHit.method},{"model",config.firstHit.modelPath.string()},
            {"profile_mode",config.firstHit.profileMode},{"profile_maximum_step",config.firstHit.profileMaximumStep},
            {"backend",config.firstHit.backend},{"batch_size",config.firstHit.batchSize},
            {"distance_convention","distance from activeDomain entry"},
            {"event","unconditioned_first_zero"},{"gradient","signed secant plus conditional SE transverse Gaussian"},
            {"sampler",{{"history_points",r.historyPoints},{"batch_size",r.batchSize},{"step_size",r.step},
                {"rank_tolerance",r.rankTolerance},{"crossing_rule",r.crossingRule}}}};
        if (config.firstHit.model) {
            result["transport"]["first_hit"]["architecture"] = config.firstHit.model->architecture();
            if (config.firstHit.model->causal()) {
                result["transport"]["first_hit"]["recurrent_step_q"] = config.firstHit.model->maximumStep();
                result["transport"]["first_hit"]["distance_sampler"] = "local_hazard_quadrature_inverse_fixed_gru_grid";
            } else if (config.firstHit.model->architecture() == "spline")
                result["transport"]["first_hit"]["distance_sampler"] = "analytic_rational_quadratic_inverse";
            else result["transport"]["first_hit"]["integration_maximum_step"] = config.firstHit.model->maximumStep();
        }
        if (config.firstHit.method == kTungstenFunctionSpaceMethod) {
            result["transport"]["first_hit"]={{"method",config.firstHit.method},{"backend","scalar"},
                {"profile_mode","tungsten_direct"},{"step_units","world"},
                {"sampler",tungstenSamplerJson(config.firstHit.tungsten)},
                {"gradient","conditioned_gaussian_with_secant_longitudinal_derivative"}};
        }
        if (config.firstHit.method == "classic_local") {
            result["transport"]["first_hit"] = {
                {"method","classic_local"},{"backend","scalar"},{"profile_mode","local_direct"},
                {"gp_model","local_tangent"},
                {"distance_convention","distance from activeDomain entry"},
                {"birth_policy","exterior_without_gp_observation"},
                {"sampler","dda_null_tracking"},{"density","scene_baked_density"},
                {"gradient","negative_flux_weighted_full_local_material_gradient"},
                {"phase_proposal","target_vndf"},
                {"surface_handoff","collision_position, F=0, full_local_gradient"}};
        }
        if (config.firstHit.method == "global_conditional" ||
            config.firstHit.method == "global_conditional_renewal_plus") {
            result["transport"]["first_hit"] = {
                {"method",config.firstHit.method},{"backend","scalar"},{"profile_mode","conditional_direct"},
                {"distance_convention","distance from activeDomain entry"},
                {"birth_policy","sampled_positive_exterior"},
                {"entry_value","F(entry) drawn from N(mean,sigma^2) conditioned on F>0"},
                {"entry_gradient","full SE Gaussian prior, independent of entry value"},
                {"entry_weight","unit: rendering is conditional on positive exterior"},
                {"sampler","segmented_delta_tracking"},
                {"majorant","analytic_birth_and_conditional_interval_bounds"},
                {"conditioning","sampled_entry_value_and_full_gradient"},
                {"gradient","negative_flux_weighted_conditional_full_gradient"}};
            if (config.firstHit.method == "global_conditional_renewal_plus") {
                auto& first = result["transport"]["first_hit"];
                first["renewal_distance"] = config.firstHit.renewalDistance;
                first["distance_units"] = "world";
                first["conditioning"] = "current_segment_value_and_full_gradient";
                first["boundary_policy"] = "sample_previous_segment_positive_endpoint";
                first["mean_policy"] = "global_field";
                first["kernel_policy"] = "shared_global_kernel";
                first["surface_handoff"] = "collision_position, F=0, full_conditional_gradient";
            }
        }
        result["transport"]["side_convention"] = config.continuationMethod == "classic_local"
            ? "first_hit preserves initial_sign; classic_local uses scene density and original field orientation"
            : "side=initial_sign; continuation uses side*mean and side*gradient";
        if (config.continuationMethod == kTungstenFunctionSpaceMethod || config.firstHit.method == kTungstenFunctionSpaceMethod)
            result["transport"]["side_convention"]="original field; reject upcrossing; no sign flip at GPIS handoff";
        result["budgets"]["write_first_hit_samples"] = config.render.writeFirstHitSamples;
    }
    result["budgets"]["safety_depth_cap"] = config.render.safetyDepthCap;
    result["budgets"]["roulette_start_depth"] = config.render.rouletteStartDepth;
    result["transport"]["classic"]["sampler"] = "dda_null_tracking";
    result["transport"]["global_conditional"] = {
        {"sampler", "two_segment_delta_tracking"},
        {"majorant", "analytic_birth_and_conditional_interval_bounds"},
        {"tracking_segments", 2},
        {"far_interval_thinning", true},
        {"birth_policy", config.transportMode == "hybrid_first_hit" && config.continuationMethod == "global_conditional"
            ? "first_hit_surface" : "sampled_positive_exterior"}};
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
    result["rng_stream"] = config.transportMode == "hybrid_first_hit"
        ? "first_hit_split_v1: seed,pixel,sample; camera=0,event=1,gradient=2,continuation=3" : "per_pixel_v1";
    if (config.transportMode == "hybrid_first_hit" &&
        (config.firstHit.method == kTungstenFunctionSpaceMethod || config.continuationMethod == kTungstenFunctionSpaceMethod)) {
        result["rng_stream"]=config.firstHit.method==kTungstenFunctionSpaceMethod
            ? "tungsten_pcg_v1: seed=event_rng.integer(), shared GPIS distance/gradient/tail stream"
            : "first_hit_split_v1 with Tungsten tail: seed=continuation_rng.integer()";
        result["transport"]["tungsten_compatibility"]={{"scope","SE standard GP, renewal+, CPU function space"},
            {"tabulated_mean","float BoxSampler, two-voxel boundary clamp, forward difference 0.001 world"},
            {"renderer","macrofacet camera, conductor and roulette; not pixel-identical Tungsten rendering"}};
    }
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write resolved config");
    stream << std::setw(2) << result << '\n';
}

} // namespace mf

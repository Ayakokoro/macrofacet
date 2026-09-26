#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <limits>

namespace {

struct TemporaryConfig {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("macrofacet_roughness_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");

    ~TemporaryConfig() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

    void write(const nlohmann::json& value) const {
        std::ofstream stream(path);
        if (!stream) throw std::runtime_error("cannot create temporary roughness config");
        stream << value.dump();
        if (!stream) throw std::runtime_error("cannot write temporary roughness config");
    }
};

struct RestoreNumericPolicy {
    mf::NumericPolicy saved = mf::defaultNumericPolicy();
    ~RestoreNumericPolicy() { mf::defaultNumericPolicy() = saved; }
};

nlohmann::json roughnessConfig() {
    return nlohmann::json::parse(R"json({
        "field": {
            "mean_type": "plane", "plane_normal": [0, 0, 1], "plane_offset": 0,
            "sigma": 0.03,
            "domain_min": [-2, -2, -0.3], "domain_max": [2, 2, 0.3]
        },
        "material": {"eta_rgb": [0.2, 0.9, 1.1], "k_rgb": [3.9, 2.5, 2.2],
                     "ndf_family": "generalized_gaussian", "roughness": 0.6},
        "transport": {},
        "fixed_flight": {
            "birth_position": [0, 0, 0], "direction": [0, 0, 1],
            "requested_maximum_age": 0.3, "curve_sample_count": 3, "flight_sample_count": 4
        },
        "numeric": {
            "relative_tolerance": 0.0001, "absolute_tolerance": 0.0000001,
            "max_quadrature_subdivisions": 128, "max_root_iterations": 64
        },
        "render": {
            "width": 2, "height": 2, "samples_per_pixel": 1,
            "camera_position": [0, -2, 1], "camera_target": [0, 0, 0],
            "vertical_fov_degrees": 45, "environment": "directional_gradient"
        },
        "output_directory": "unused_roughness_test_output"
    })json");
}

} // namespace

void testMaterialRoughness(TestContext& context) {
    using namespace mf;
    RestoreNumericPolicy restorePolicy;
    auto defaultConfig = [] {
        ExperimentConfig config;
        config.field = buildDefaultField();
        return config;
    };
    auto checkRoughness = [&](const ExperimentConfig& config, double sigma, double roughness) {
        const PointPrior prior = config.field.pointPrior(Point3::Zero());
        context.near(prior.varianceF, sigma * sigma, 1e-14,
                     "material roughness keeps the requested field amplitude");
        context.require((prior.covarianceG - roughness * roughness * Matrix3::Identity()).norm() < 1e-12,
                        "material roughness is the isotropic gradient standard deviation");
        const double correlationLength = sigma / roughness;
        for (int axis = 0; axis < 3; ++axis) {
            const KernelJet jet = config.field.kernel.evaluate(
                Point3::Zero(), correlationLength * Vector3::Unit(axis));
            context.near(jet.valueValue / prior.varianceF, std::exp(-0.5), 1e-12,
                         "material roughness gives correlation length sigma / roughness");
            context.require(jet.gradientXValueY.allFinite() && jet.valueXGradientY.allFinite() &&
                                jet.gradientXGradientY.allFinite(),
                            "roughness-derived kernel has finite covariance derivatives");
        }
        context.require(config.materialRoughness.has_value(), "explicit material roughness is recorded");
        if (config.materialRoughness) {
            context.near(*config.materialRoughness, roughness, 1e-14,
                         "recorded material roughness follows the effective setting");
        }
    };

    ExperimentConfig explicitRoughness = defaultConfig();
    applyFieldOverrides(explicitRoughness, std::nullopt, 0.65, false);
    checkRoughness(explicitRoughness, 0.1, 0.65);
    applyFieldOverrides(explicitRoughness, 0.025, std::nullopt, false);
    checkRoughness(explicitRoughness, 0.025, 0.65);
    applyFieldOverrides(explicitRoughness, 0.04, 0.8, true);
    checkRoughness(explicitRoughness, 0.04, 0.8);

    // The resulting derivative covariance must agree with derivatives of the kernel.
    const Point3 x(0.01, -0.02, 0.005);
    const Point3 y(-0.01, 0.005, 0.02);
    const auto& kernel = explicitRoughness.field.kernel;
    const KernelJet jet = kernel.evaluate(x, y);
    constexpr double step = 1e-6;
    for (int axis = 0; axis < 3; ++axis) {
        const Vector3 offset = step * Vector3::Unit(axis);
        const double firstDerivative = (kernel.evaluate(x + offset, y).valueValue -
                                        kernel.evaluate(x - offset, y).valueValue) / (2.0 * step);
        context.near(jet.gradientXValueY[axis], firstDerivative, 1e-9,
                     "roughness-derived kernel first derivative matches finite differences");
        const Vector3 mixedDerivative = (kernel.evaluate(x, y + offset).gradientXValueY -
                                          kernel.evaluate(x, y - offset).gradientXValueY) / (2.0 * step);
        context.require((jet.gradientXGradientY.col(axis) - mixedDerivative).norm() < 1e-8,
                        "roughness-derived mixed covariance matches finite differences");
    }

    ExperimentConfig anisotropic = defaultConfig();
    Matrix3 rotation;
    rotation << 0.8, -0.6, 0.0, 0.6, 0.8, 0.0, 0.0, 0.0, 1.0;
    anisotropic.field.kernel = SquaredExponentialKernel::fromCorrelationLengths(
        0.12, Vector3(0.2, 0.35, 0.5), rotation);
    const Matrix3 originalPrecision = anisotropic.field.kernel.precision();
    const Matrix3 originalCovariance = anisotropic.field.pointPrior(Point3::Zero()).covarianceG;
    ExperimentConfig legacySigma = anisotropic;
    applyFieldOverrides(legacySigma, 0.03, std::nullopt, false);
    context.require((legacySigma.field.kernel.precision() - originalPrecision).norm() < 1e-12,
                    "legacy sigma override preserves anisotropic correlation lengths");
    context.require((legacySigma.field.pointPrior(Point3::Zero()).covarianceG -
                     originalCovariance / 16.0).norm() < 1e-12,
                    "legacy sigma override scales slope covariance with sigma squared");
    context.require(!legacySigma.materialRoughness.has_value(),
                    "legacy anisotropic config remains without scalar material roughness");
    ExperimentConfig preserveSlope = anisotropic;
    applyFieldOverrides(preserveSlope, 0.03, std::nullopt, true);
    context.require((preserveSlope.field.pointPrior(Point3::Zero()).covarianceG -
                     originalCovariance).norm() < 1e-12,
                    "preserve-slope retains the full rotated anisotropic covariance");
    applyFieldOverrides(anisotropic, 0.02, 0.45, true);
    checkRoughness(anisotropic, 0.02, 0.45);
    ExperimentConfig largeRoughness = defaultConfig();
    applyFieldOverrides(largeRoughness, std::nullopt, 1.5, false);
    checkRoughness(largeRoughness, 0.1, 1.5);

    for (double invalid : std::array<double, 5>{{0.0, -0.1,
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()}}) {
        bool rejected = false;
        try {
            ExperimentConfig config = defaultConfig();
            applyFieldOverrides(config, std::nullopt, invalid, false);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "roughness override rejects nonpositive or nonfinite roughness");
        rejected = false;
        try {
            ExperimentConfig config = defaultConfig();
            applyFieldOverrides(config, invalid, 0.6, false);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "roughness-derived kernel rejects nonpositive or nonfinite sigma");
    }
    for (NdfFamily family : {NdfFamily::GGXBaseline, NdfFamily::BeckmannLimit}) {
        bool rejected = false;
        try {
            ExperimentConfig config = defaultConfig();
            config.material.ndfFamily = family;
            applyFieldOverrides(config, std::nullopt, 0.6, false);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "scalar material roughness requires generalized Gaussian NDF");
    }
    for (const Vector2& parameters : std::array<Vector2, 3>{{
             Vector2(0.03, 1e200), Vector2(1e-200, 0.3), Vector2(1e100, 1e-100)}}) {
        bool rejected = false;
        try {
            ExperimentConfig config = defaultConfig();
            applyFieldOverrides(config, parameters.x(), parameters.y(), false);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "roughness parameters reject unrepresentable variances or precision");
    }

    TemporaryConfig temporary;
    const nlohmann::json valid = roughnessConfig();
    temporary.write(valid);
    ExperimentConfig loaded = loadExperimentConfig(temporary.path);
    context.require(loaded.material.ndfFamily == NdfFamily::GeneralizedGaussian,
                    "NDF family is parsed from the material block");
    checkRoughness(loaded, 0.03, 0.6);
    applyFieldOverrides(loaded, 0.01, std::nullopt, false);
    checkRoughness(loaded, 0.01, 0.6);
    applyFieldOverrides(loaded, std::nullopt, 0.9, false);
    checkRoughness(loaded, 0.01, 0.9);
    writeResolvedConfig(loaded, temporary.path);
    {
        std::ifstream stream(temporary.path);
        nlohmann::json resolved;
        stream >> resolved;
        context.near(resolved.at("material").at("roughness").get<double>(), 0.9, 1e-14,
                     "resolved metadata records the override instead of the input roughness");
        context.require(resolved.at("material").at("ndf_family") == "generalized_gaussian",
                        "resolved metadata records the material NDF family");
        context.near(resolved.at("derived").at("isotropic_correlation_length").get<double>(),
                     0.01 / 0.9, 1e-14, "resolved metadata records effective correlation length");
        for (const auto& component : resolved.at("derived").at("gradient_stddev_xyz")) {
            context.near(component.get<double>(), 0.9, 1e-14,
                         "resolved metadata agrees with effective isotropic gradient statistics");
        }
    }

    auto rejectsJson = [&](const nlohmann::json& invalid, const std::string& message) {
        temporary.write(invalid);
        bool rejected = false;
        try {
            (void)loadExperimentConfig(temporary.path);
        } catch (const std::exception&) {
            rejected = true;
        }
        context.require(rejected, message);
    };
    nlohmann::json ambiguous = valid;
    ambiguous["field"]["correlation_lengths"] = {0.1, 0.1, 0.1};
    rejectsJson(ambiguous, "JSON material roughness cannot be combined with correlation lengths");
    for (const nlohmann::json& value : std::array<nlohmann::json, 7>{{
             "0.6", nlohmann::json::array({0.6, 0.6}), nullptr, 0.0, -0.1, true, false}}) {
        nlohmann::json invalid = valid;
        invalid["material"]["roughness"] = value;
        rejectsJson(invalid, "JSON rejects nonscalar or nonpositive roughness");
    }
    for (const char* family : {"ggx", "beckmann_limit"}) {
        nlohmann::json invalid = valid;
        invalid["material"]["ndf_family"] = family;
        rejectsJson(invalid, "JSON roughness rejects non-generalized-Gaussian families");
    }
    nlohmann::json ggx = valid;
    ggx["material"].erase("roughness");
    ggx["material"]["ndf_family"] = "ggx";
    ggx["material"]["ggx_alpha"] = {0.3, 0.5};
    ggx["field"]["correlation_lengths"] = {0.1, 0.1, 0.1};
    temporary.write(ggx);
    const ExperimentConfig classicGgx = loadExperimentConfig(temporary.path);
    context.require(classicGgx.material.ndfFamily == NdfFamily::GGXBaseline &&
                    (classicGgx.material.ggxAlpha - Vector2(0.3, 0.5)).norm() < 1e-12,
                    "Classic reads GGX selection and alpha from material");
    ggx["modes"] = {"conditional29"};
    rejectsJson(ggx, "obsolete mode selection is rejected");
    ggx.erase("modes");
    ggx["transport"]["conditional29"] = {{"sampler", "regular_tracking"}};
    rejectsJson(ggx, "obsolete conditional transport settings are rejected");
}

#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
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

class UnitDensity final : public mf::ScalarField {
public:
    double sample(const mf::Point3&) const override { return 1.0; }
    mf::ScalarBounds bounds(const mf::Bounds3&) const override {
        return {1.0, 1.0, true};
    }
};

bool sameRender(const mf::RenderedImage& a, const mf::RenderedImage& b) {
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size())
        return false;
    for (std::size_t i = 0; i < a.pixels.size(); ++i) {
        if (!(a.pixels[i].array() == b.pixels[i].array()).all()) return false;
    }
    const mf::RenderStatistics& x = a.statistics;
    const mf::RenderStatistics& y = b.statistics;
    return x.paths == y.paths && x.realCollisions == y.realCollisions &&
           x.escapedPaths == y.escapedPaths &&
           x.rouletteTerminations == y.rouletteTerminations &&
           x.safetyCapTerminations == y.safetyCapTerminations &&
           x.numericalFailures == y.numericalFailures &&
           x.accumulatedPathDepth == y.accumulatedPathDepth &&
           x.tracking.candidates == y.tracking.candidates &&
           x.tracking.nullCollisions == y.tracking.nullCollisions;
}

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
        context.require((prior.covarianceG - 0.5 * roughness * roughness * Matrix3::Identity()).norm() < 1e-12,
                        "material roughness is Beckmann alpha, so gradient covariance is alpha squared over two");
        context.near(std::sqrt(2.0 * prior.covarianceG(0, 0)), roughness, 1e-12,
                     "the NDF's transverse variance recovers the requested Beckmann alpha");
        context.require(!GaussianNdf(prior.meanG, prior.covarianceG).isBeckmannLimit(),
                        "a finite longitudinal variance keeps the generalized Gaussian distinct from Beckmann");
        const Point3 offSurface(0.0, 0.0, 0.1);
        const PointPrior global = config.field.pointPrior(offSurface);
        const PointPrior local = config.material.materialNdf(config.field, offSurface);
        context.near(local.meanF, 0.0, 1e-14,
                     "the local material GP is centred on the tangent plane");
        context.require((local.meanG - Vector3::UnitZ()).norm() < 1e-14,
                        "the local material GP has a unit normal mean gradient");
        context.near(local.varianceF, global.varianceF, 1e-14,
                     "the local material GP uses the prescribed sigma");
        context.require((local.covarianceG - global.covarianceG).norm() < 1e-12,
                        "without an alpha field the local roughness equals the global GP roughness");
        const double correlationLength = 1.0 / std::sqrt(config.field.kernel.metric()(0, 0));
        double expectedCorrelation = std::exp(-0.5);
        if (config.field.kernel.type() == CovarianceKernelType::Matern32) {
            const double a = 1.0;
            expectedCorrelation = (1.0 + a) * std::exp(-a);
        } else if (config.field.kernel.type() == CovarianceKernelType::Matern52) {
            const double a = std::sqrt(5.0);
            expectedCorrelation = (1.0 + a + 5.0 / 3.0) * std::exp(-a);
        }
        for (int axis = 0; axis < 3; ++axis) {
            const KernelJet jet = config.field.kernel.evaluate(
                Point3::Zero(), correlationLength * Vector3::Unit(axis));
            context.near(jet.valueValue / prior.varianceF, expectedCorrelation, 1e-12,
                         "kernel-specific correlation length preserves material roughness");
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

    // All three families share exactly the same one-point value/gradient law
    // when configured by material roughness. Classic transport must therefore
    // remain independent of the selected spatial covariance family.
    std::array<ExperimentConfig, 3> familyConfigs;
    const std::array<const char*, 3> familyNames{
        "squared_exponential", "matern_3_2", "matern_5_2"};
    const std::array<double, 3> lengthFactors{
        1.0, 1.0, std::sqrt(5.0 / 3.0)};
    for (std::size_t family = 0; family < familyConfigs.size(); ++family) {
        nlohmann::json source = valid;
        source["field"]["kernel_type"] = familyNames[family];
        temporary.write(source);
        familyConfigs[family] = loadExperimentConfig(temporary.path);
        checkRoughness(familyConfigs[family], 0.03, 0.6);
        context.near(1.0 / std::sqrt(familyConfigs[family].field.kernel.metric()(0, 0)),
                     lengthFactors[family] * 0.03 / (0.6 / std::sqrt(2.0)), 1e-14,
                     "roughness converts to the family-specific correlation length");
    }

    const auto density = std::make_shared<UnitDensity>();
    const auto densityGrid = std::make_shared<DensityMajorantGrid>(
        *density, familyConfigs[0].field.activeDomain, 4);
    for (ExperimentConfig& config : familyConfigs) {
        config.mediumDensity = density;
        config.densityMajorantGrid = densityGrid;
        config.render.width = 4;
        config.render.height = 3;
        config.render.samplesPerPixel = 8;
        config.render.threadCount = 1;
        config.render.safetyDepthCap = 8;
        config.render.rouletteStartDepth = 5;
        config.classicPhaseProposal = "target_vndf";
        config.preparedAreaMajorant.reset();
    }
    for (const GpModel model : {GpModel::LocalTangent, GpModel::GlobalPointwise}) {
        std::array<RenderedImage, 3> renders;
        for (std::size_t family = 0; family < familyConfigs.size(); ++family) {
            familyConfigs[family].material.gpModel = model;
            familyConfigs[family].transportMode = model == GpModel::LocalTangent
                ? "classic_local" : "classic_global";
            renders[family] = renderAnalyticScene(familyConfigs[family]);
        }
        context.require(sameRender(renders[0], renders[1]),
                        "SE and Matern 3/2 render identically in a roughness-matched classic mode");
        context.require(sameRender(renders[0], renders[2]),
                        "SE and Matern 5/2 render identically in a roughness-matched classic mode");
    }
    nlohmann::json globalJson = valid;
    globalJson["material"]["gp_model"] = "global_pointwise";
    temporary.write(globalJson);
    const ExperimentConfig globalLoaded = loadExperimentConfig(temporary.path);
    context.require(globalLoaded.material.gpModel == GpModel::GlobalPointwise,
                    "global pointwise GP model is parsed from material");
    const Point3 globalPoint(0.0, 0.0, 0.1);
    const PointPrior globalPrior = globalLoaded.field.pointPrior(globalPoint);
    const PointPrior globalMaterial = globalLoaded.material.materialNdf(globalLoaded.field, globalPoint);
    context.near(globalMaterial.meanF, globalPrior.meanF, 1e-14,
                 "global material retains the field mean value at the point");
    context.require((globalMaterial.meanG - globalPrior.meanG).norm() < 1e-14 &&
                    (globalMaterial.covarianceG - globalPrior.covarianceG).norm() < 1e-14,
                    "global material reads the field's gradient statistics without a local replacement");
    const Point3 surfacePoint = Point3::Zero();
    const Vector3 incidence = Vector3(0.3, 0.4, -0.5).normalized();
    context.near(classicProjectedArea(globalLoaded.field, globalLoaded.material,
                                      surfacePoint, incidence).value,
                 classicProjectedArea(loaded.field, loaded.material,
                                      surfacePoint, incidence).value, 1e-12,
                 "global and local models agree for a planar unit-gradient mean and equal alpha");
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
        context.require(resolved.at("material").at("gp_model") == "local_tangent",
                        "resolved metadata records the default local GP model");
        context.require(resolved.at("derived").at("kernel_type") == "squared_exponential",
                        "resolved metadata records the covariance kernel family");
        context.near(resolved.at("derived").at("isotropic_correlation_length").get<double>(),
                     0.01 / (0.9 / std::sqrt(2.0)), 1e-14,
                     "resolved metadata records effective correlation length");
        for (const auto& component : resolved.at("derived").at("gradient_stddev_xyz")) {
            context.near(component.get<double>(), 0.9 / std::sqrt(2.0), 1e-14,
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
    nlohmann::json unknownModel = valid;
    unknownModel["material"]["gp_model"] = "unknown";
    rejectsJson(unknownModel, "unknown GP model is rejected");
    nlohmann::json unknownKernel = valid;
    unknownKernel["field"]["kernel_type"] = "matern_7_2";
    rejectsJson(unknownKernel, "unknown covariance kernel is rejected");
    nlohmann::json globalGgx = globalJson;
    globalGgx["material"].erase("roughness");
    globalGgx["material"]["ndf_family"] = "ggx";
    globalGgx["field"]["correlation_lengths"] = {0.1, 0.1, 0.1};
    rejectsJson(globalGgx, "global pointwise mode rejects GGX because it needs GP gradient statistics");
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

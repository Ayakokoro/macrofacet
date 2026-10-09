#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/macrofacet/MaterialConfig.h"
#include "macrofacet/mathutility/NumericPolicy.h"
#include "macrofacet/transport/FlightState.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mf {

class RenewalHazardModel;

struct RenewalRenderConfig {
    std::filesystem::path modelPath;
    double profileMaximumStep = 0.25;
    std::string profileMode = "cubic"; // cubic (exact cells), point_linear (lazy tangent approximation)
    std::string backend = "scalar"; // scalar, torch_cpu, torch_cuda, auto
    std::string resolvedBackend = "scalar";
    int batchSize = 4096;
    // Loaded once, immutable and shared by every render worker.
    std::shared_ptr<const RenewalHazardModel> model;
};

struct RenderConfig {
    int width = 64;
    int height = 64;
    int samplesPerPixel = 256;
    Point3 cameraPosition = Point3(0.0, 0.0, 1.0);
    Point3 cameraTarget = Point3::Zero();
    double verticalFovDegrees = 45.0;
    std::string environment = "directional_gradient";
    int rouletteStartDepth = 5;
    int safetyDepthCap = 64;
    int threadCount = 0;  // 0 selects the hardware concurrency
};

struct ConditionalBirthConfig {
    // sample_positive_exterior matches camera-ray initialization. fixed_observation
    // is useful for comparing flights that share an exact GP history.
    std::string policy = "sample_positive_exterior";
    double value = 0.0;
    Vector3 gradient = Vector3::Zero();
};

struct TransmittanceRayConfig {
    std::string id;
    std::string group = "ungrouped";
    Point3 origin = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    double maximumDistance = 0.0;
    std::optional<int> bins;
    std::optional<ConditionalBirthConfig> conditionalBirth;
};

struct TransmittanceConfig {
    std::vector<std::string> modes{
        "classic_local", "classic_global", "global_conditional"};
    int bins = 256;
    std::string spacing = "linear";
    std::string distanceOrigin = "ray_origin";
    int trialsPerRay = 16384;
    double confidenceLevel = 0.95;
    bool writeRawSamples = false;
    ConditionalBirthConfig conditionalBirth;
    std::string localTransportReference = "classic_global";
    bool constantExponentialEnabled = true;
    std::string constantExponentialFit = "censored_mle";
    std::vector<TransmittanceRayConfig> rays;
};

struct ExperimentConfig {
    int schemaVersion = 1;
    std::uint64_t seed = 17429;
    // Mean field, covariance kernel, and active domain shared by both GP models.
    // global-pointwise mode uses the mean field directly; local-tangent mode builds a
    // tangent-plane GP at each point from the mean's gradient and Hessian.
    GPSSField field;
    // Baked density and its certified majorant drive DDA tracking.
    ScalarFieldPtr mediumDensity;
    std::shared_ptr<const DensityMajorantGrid> densityMajorantGrid;
    // Certified projected-area bound for the selected GP model, prepared once.
    std::optional<double> preparedAreaMajorant;
    MaterialConfig material;
    // Original field description, used to identify the cached analytic bake.
    std::string sourceFieldSpec;
    std::optional<double> bakeVoxelSize;
    // Explicit scalar Beckmann alpha, if configured. It fixes
    // covarianceG = roughness^2 / 2 I; the kernel family determines the
    // correlation length required to realize that same pointwise roughness.
    // In local mode only, an alpha grid overrides this covariance per point.
    std::optional<double> materialRoughness;
    RenderConfig render;
    std::optional<TransmittanceConfig> transmittance;
    NumericPolicy numeric;
    std::filesystem::path outputDirectory = "outputs/macrofacet_experiments";
    double beckmannMixtureWeight = 0.5;
    std::string classicPhaseProposal = "uniform";
    // classic obeys material.gp_model; comparison modes select it explicitly.
    std::string transportMode = "classic";
    RenewalRenderConfig renewal;
};

ExperimentConfig loadExperimentConfig(const std::filesystem::path& path);
void prepareRenewalModel(ExperimentConfig& config);
void requireNanoVdbField(const ExperimentConfig& config);
void applyFieldOverrides(ExperimentConfig& config, std::optional<double> sigma,
                         std::optional<double> roughness, bool preserveSlope);
void writeResolvedConfig(const ExperimentConfig& config, const std::filesystem::path& path);
GPSSField buildDefaultField();

} // namespace mf

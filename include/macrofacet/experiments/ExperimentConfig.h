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

namespace mf {

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

struct ExperimentConfig {
    int schemaVersion = 1;
    std::uint64_t seed = 17429;
    GPSSField field;
    // Baked density and its certified majorant drive DDA tracking.
    ScalarFieldPtr mediumDensity;
    std::shared_ptr<const DensityMajorantGrid> densityMajorantGrid;
    // Certified local projected-area bound baked once during field preparation.
    std::optional<double> preparedAreaMajorant;
    MaterialConfig material;
    // Original field description, used to identify the cached analytic bake.
    std::string sourceFieldSpec;
    std::optional<double> bakeVoxelSize;
    // Default local GP roughness in the Beckmann-alpha convention. It fixes
    // covarianceG = roughness^2 / 2 I and ell = sqrt(2) * sigma / roughness.
    // An alpha grid overrides this covariance at each material point.
    std::optional<double> materialRoughness;
    RenderConfig render;
    NumericPolicy numeric;
    std::filesystem::path outputDirectory = "outputs/macrofacet_experiments";
    double beckmannMixtureWeight = 0.5;
    std::string classicPhaseProposal = "uniform";
};

ExperimentConfig loadExperimentConfig(const std::filesystem::path& path);
void requireNanoVdbField(const ExperimentConfig& config);
void applyFieldOverrides(ExperimentConfig& config, std::optional<double> sigma,
                         std::optional<double> roughness, bool preserveSlope);
void writeResolvedConfig(const ExperimentConfig& config, const std::filesystem::path& path);
GPSSField buildDefaultField();

} // namespace mf

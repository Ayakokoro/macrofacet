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

struct FixedFlightConfig {
    Point3 birthPosition = Point3::Zero();
    Vector3 direction = Vector3(0.984807753012208, 0.0, 0.17364817766693033);
    double requestedMaximumAge = 1.5;
    int curveSampleCount = 65;
    int flightSampleCount = 20000;
};

struct RenderConfig {
    int width = 64;
    int height = 64;
    int samplesPerPixel = 256;
    Point3 cameraPosition = Point3(0.0, 0.0, 1.0);
    Point3 cameraTarget = Point3::Zero();
    double verticalFovDegrees = 45.0;
    std::string environment = "directional_gradient";
    int flightTableCells = 48;
    int rouletteStartDepth = 5;
    int safetyDepthCap = 64;
    int threadCount = 0;  // 0 selects the hardware concurrency
};

struct ExperimentConfig {
    int schemaVersion = 1;
    std::uint64_t seed = 17429;
    GPSSField field;
    // Baked density defines the transport band; null is for core-only analytic tests.
    ScalarFieldPtr mediumDensity;
    bool mediumSurfaceBand = false;
    std::shared_ptr<const DensityMajorantGrid> densityMajorantGrid;
    MaterialConfig material;
    // Original field description, used to identify the cached analytic bake.
    std::string sourceFieldSpec;
    std::optional<double> bakeVoxelSize;
    // Isotropic Gaussian gradient-component standard deviation, not GGX alpha
    // or a perceptual roughness mapping. When present, ell = sigma / roughness.
    std::optional<double> materialRoughness;
    FixedFlightConfig fixedFlight;
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

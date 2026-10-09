#pragma once

#include "macrofacet/gpss/MeanField.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mf {

struct FirstPassageKernelConfig {
    std::string id;
    std::string type = "squared_exponential";
    double variance = 1.0;
    double lengthScale = 1.0;
    // Used only by rational_quadratic.
    double alpha = 1.0;
};

struct FirstPassageStateAnalysisConfig {
    bool enabled = true;
    std::vector<double> snapshotTimes{0.5, 1.0, 2.0, 4.0, 8.0};
    double futureWindow = 0.05;
    int valueBins = 16;
    int derivativeBins = 16;
    std::optional<double> valueMinimum;
    std::optional<double> valueMaximum;
    std::optional<double> derivativeMinimum;
    std::optional<double> derivativeMaximum;
    bool writeSamples = false;
};

struct FirstPassageRiceSeriesConfig {
    bool enabled = true;
    // This implementation currently supports the first two Rice terms.
    int maxOrder = 2;
    double relativeTolerance = 1e-6;
    double absoluteTolerance = 1e-9;
    int maximumQuadratureSubdivisions = 1024;
    // Smooth kernels make the two-time conditioning nearly rank deficient as
    // the crossing times coalesce. This only controls PSD roundoff clipping.
    double covarianceRoundoffMultiplier = 256.0;
};

struct FirstPassageRayCondition {
    Point3 origin = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    // Unnormalized observed gradient of F in surface mode; unused in exterior mode.
    Vector3 gradient = Vector3::UnitZ();
};

struct FirstPassageCollisionState {
    std::string id;
    double beta0 = 0.0;
    double betaMeanSlope = 0.0;
    double betaCollisionSlope = 1.0;
    std::optional<FirstPassageRayCondition> ray;
};

struct FirstPassageCollisionSamplerConfig {
    double minimumStep = 0.00125;
    double crossingTolerance = 1e-8;
    double bridgeSigmaMargin = 6.0;
    int maximumRefinementDepth = 12;
};

struct FirstPassageFixedEndpointConfig {
    bool enabled = false;
    // Skip ordinary FPT generation; write only fixed-endpoint proposals/statistics.
    bool only = false;
    // Physical scene distances, converted to q independently for each kernel.
    std::vector<double> distances;
    // 0 reuses monte_carlo.trajectories (and the CLI --trials override).
    int trajectories = 0;
};

struct FirstPassageExperimentConfig {
    int schemaVersion = 1;
    std::uint64_t seed = 17429;
    double processMean = 0.0;
    // Optional full spatial mean. Physical ray conditions replace beta inputs.
    MeanFieldPtr processMeanField;
    std::string meanFieldConfiguration;
    double threshold = 0.0;
    std::string initialConditionType = "fixed_value";
    double initialValue = 1.0;
    double maximumTime = 10.0;
    // Geometry/network partition, independent of GP reference resolution.
    double profileMaximumStep = 0.25;
    // fixed_value: the smallest entry is sampled and larger entries are coupled
    // subsamples. collision_state/positive_exterior: each entry is an independent
    // backend-specific convergence resolution; the smallest is the training set.
    std::vector<double> stepSizes{0.04, 0.02, 0.01};
    int curveBins = 200;
    int trajectories = 16384;
    int threadCount = 0; // 0 selects hardware concurrency
    double confidenceLevel = 0.95;
    bool writeRawSamples = false;
    int maximumEmbeddingExpansions = 8;
    int minimumRiskSetForError = 32;
    FirstPassageStateAnalysisConfig stateAnalysis;
    FirstPassageRiceSeriesConfig riceSeries;
    FirstPassageCollisionSamplerConfig collisionSampler;
    FirstPassageFixedEndpointConfig fixedEndpoint;
    std::vector<FirstPassageCollisionState> collisionStates;
    std::vector<FirstPassageKernelConfig> kernels;
    std::filesystem::path outputDirectory = "outputs/first_passage";
};

FirstPassageExperimentConfig loadFirstPassageExperimentConfig(
    const std::filesystem::path& path);
void writeResolvedFirstPassageConfig(const FirstPassageExperimentConfig& config,
                                     const std::filesystem::path& path);
void runFirstPassageExperiment(const FirstPassageExperimentConfig& config);

} // namespace mf

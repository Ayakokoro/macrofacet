#pragma once

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

struct FirstPassageExperimentConfig {
    int schemaVersion = 1;
    std::uint64_t seed = 17429;
    double processMean = 0.0;
    double threshold = 0.0;
    double initialValue = 1.0;
    double maximumTime = 10.0;
    // The smallest entry is sampled. Larger entries must be integer multiples
    // and are coupled subsamples used for crossing-resolution convergence.
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
    std::vector<FirstPassageKernelConfig> kernels;
    std::filesystem::path outputDirectory = "outputs/first_passage";
};

FirstPassageExperimentConfig loadFirstPassageExperimentConfig(
    const std::filesystem::path& path);
void writeResolvedFirstPassageConfig(const FirstPassageExperimentConfig& config,
                                     const std::filesystem::path& path);
void runFirstPassageExperiment(const FirstPassageExperimentConfig& config);

} // namespace mf

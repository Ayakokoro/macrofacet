#include "TestHarness.h"
#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/FirstPassageRice.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void testFirstPassage(TestContext& context) {
    using namespace mf;
    namespace fs = std::filesystem;
    const std::vector<FirstPassageKernelConfig> derivativeKernels{
        {"se", "squared_exponential", 1.3, 0.8, 1.0},
        {"m32", "matern_3_2", 1.3, 0.8, 1.0},
        {"m52", "matern_5_2", 1.3, 0.8, 1.0},
        {"rq", "rational_quadratic", 1.3, 0.8, 2.5}};
    for (const FirstPassageKernelConfig& source : derivativeKernels) {
        const FirstPassageStationaryKernel kernel(source);
        const double r = 0.37;
        const double h = 1e-5;
        const double finiteFirst =
            (kernel.covariance(r + h) - kernel.covariance(r - h)) / (2.0 * h);
        const double finiteSecond =
            (kernel.covariance(r + h) - 2.0 * kernel.covariance(r) +
             kernel.covariance(r - h)) / (h * h);
        context.relative(kernel.firstDerivative(r), finiteFirst, 2e-8,
                         source.id + " covariance first derivative");
        context.relative(kernel.secondDerivative(r), finiteSecond, 2e-5,
                         source.id + " covariance second derivative");
        context.near(kernel.derivativeVariance(), -kernel.secondDerivative(0.0),
                     1e-14, source.id + " derivative variance identity");
    }

    const FirstPassageStationaryKernel se(
        {"se", "squared_exponential", 1.0, 1.0, 1.0});
    const double queryTime = 0.4;
    const double c = std::exp(-0.5 * queryTime * queryTime);
    const double cp = -queryTime * c;
    const DynamicGaussian conditioned = conditionValueDerivativeGaussian(
        se, {queryTime}, 0.0, {0.0}, {1.0});
    context.near(conditioned.mean[0], c, 1e-13,
                 "joint conditioner gives conditioned value mean");
    context.near(conditioned.mean[1], cp, 1e-13,
                 "joint conditioner gives conditioned derivative mean");
    context.near(conditioned.covariance(0, 0), 1.0 - c * c, 1e-13,
                 "joint conditioner gives conditioned value variance");
    context.near(conditioned.covariance(1, 1), 1.0 - cp * cp, 1e-13,
                 "joint conditioner gives conditioned derivative variance");
    context.near(conditioned.covariance(0, 1), -c * cp, 1e-13,
                 "joint conditioner gives value-derivative covariance");
    context.require(riceDowncrossingW1(se, 0.0, 0.0, 1.0, 0.8) > 0.0,
                    "Rice W1 is a positive boundary flux");
    context.require(riceDowncrossingW2(se, 0.0, 0.0, 1.0, 0.5, 1.0) >= 0.0,
                    "Rice W2 is a nonnegative product intensity");

    const fs::path directory = fs::temp_directory_path() / "macrofacet_first_passage_test";
    const fs::path input = directory / "input.json";
    fs::create_directories(directory);
    nlohmann::json root = {
        {"schema_version", 1},
        {"seed", 1234},
        {"first_passage", {
            {"process", {{"mean", 0.0}, {"threshold", -100.0}}},
            {"initial_condition", {{"type", "fixed_value"}, {"value", 1.0}}},
            {"grid", {{"max_time", 1.0}, {"step_sizes", {0.2, 0.1}}}},
            {"curve", {{"bins", 10}}},
            {"rice_series", {{"enabled", true}, {"max_order", 2},
                               {"relative_tolerance", 1e-5},
                               {"absolute_tolerance", 1e-8},
                               {"max_quadrature_subdivisions", 128}}},
            {"monte_carlo", {{"trajectories", 2048},
                               {"confidence_level", 0.9},
                               {"minimum_risk_set_for_error", 2}}},
            {"state_analysis", {{"enabled", true},
                                  {"snapshot_times", {0.4}},
                                  {"future_window", 0.1},
                                  {"value_bins", 4},
                                  {"derivative_bins", 4}}},
            {"kernels", {
                {{"id", "se"}, {"type", "squared_exponential"},
                 {"variance", 1.0}, {"length_scale", 1.0}},
                {{"id", "m52"}, {"type", "matern_5_2"},
                 {"variance", 0.5}, {"length_scale", 0.8}}}}}},
        {"output_directory", directory.string()}};
    {
        std::ofstream stream(input);
        stream << root;
    }

    FirstPassageExperimentConfig config = loadFirstPassageExperimentConfig(input);
    context.require(config.kernels.size() == 2 && config.stepSizes.size() == 2,
                    "first-passage kernels and convergence steps are parsed");
    context.near(config.confidenceLevel, 0.9, 0.0,
                 "first-passage confidence level is parsed");
    runFirstPassageExperiment(config);
    context.require(fs::exists(directory / "first_passage_curves.csv") &&
                    fs::exists(directory / "first_passage_summary.csv") &&
                    fs::exists(directory / "first_passage_state_hazard.csv") &&
                    fs::exists(directory / "first_passage_hazard.svg") &&
                    fs::exists(directory / "first_passage_survival.svg") &&
                    fs::exists(directory / "first_passage_rice_density.svg") &&
                    fs::exists(directory / "resolved_first_passage_config.json"),
                    "first-passage experiment writes curve, state, plot, and provenance outputs");
    std::ifstream curves(directory / "first_passage_curves.csv");
    std::string header;
    std::getline(curves, header);
    context.require(header.find("hazard_mc") != std::string::npos &&
                    header.find("endpoint_conditioned_hazard_sigma1") != std::string::npos &&
                    header.find("pointwise_hazard_sigma2") != std::string::npos &&
                    header.find("start_conditioned_endpoint_survival") != std::string::npos &&
                    header.find("start_conditioned_sigma1_survival") != std::string::npos &&
                    header.find("rice_w1") != std::string::npos &&
                    header.find("rice_w2_integral") != std::string::npos &&
                    header.find("rice_density_order2") != std::string::npos,
                    "first-passage curve separates exact and start-conditioned approximations");
    curves.close();
    std::ifstream stateSummary(directory / "first_passage_state_summary.csv");
    std::string stateHeader;
    std::string stateRow;
    std::getline(stateSummary, stateHeader);
    std::getline(stateSummary, stateRow);
    std::vector<std::string> fields;
    std::stringstream rowStream(stateRow);
    for (std::string field; std::getline(rowStream, field, ',');) fields.push_back(field);
    const double sampledC = std::exp(-0.5 * 0.4 * 0.4);
    context.require(fields.size() >= 10,
                    "first-passage state summary has all moment columns");
    if (fields.size() >= 10) {
        context.near(std::stod(fields[8]), sampledC, 0.04,
                     "circulant sampler has the conditioned SE mean");
        context.near(std::stod(fields[9]), std::sqrt(1.0 - sampledC * sampledC), 0.04,
                     "circulant sampler has the conditioned SE standard deviation");
    }
    stateSummary.close();
    fs::remove_all(directory);
}

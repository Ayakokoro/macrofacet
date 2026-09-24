#include "macrofacet/experiments/ConditionalGPReference.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/experiments/FlightCurveExperiment.h"
#include "macrofacet/experiments/RenderExperiment.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct CommandLine {
    std::string command;
    std::filesystem::path configPath;
    std::optional<double> sigma;
    std::optional<double> roughness;
    std::optional<std::filesystem::path> outputDirectory;
    std::optional<int> samplesPerPixel;
    std::optional<int> flightTableCells;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> threadCount;
    bool preserveSlope = false;
};

int parsePositiveInteger(const std::string& option, const std::string& value) {
    std::size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result < 1) {
        throw std::invalid_argument(option + " must be a positive integer");
    }
    return result;
}

int parseNonnegativeInteger(const std::string& option, const std::string& value) {
    std::size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result < 0) {
        throw std::invalid_argument(option + " must be a nonnegative integer");
    }
    return result;
}

double parsePositiveNumber(const std::string& option, const std::string& value) {
    std::size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !(result > 0.0) || !std::isfinite(result)) {
        throw std::invalid_argument(option + " must be a finite positive number");
    }
    return result;
}

CommandLine parseCommandLine(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("missing command");
    CommandLine options;
    options.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--preserve-slope") {
            options.preserveSlope = true;
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + argument);
        const std::string value = argv[++i];
        if (argument == "--config") options.configPath = value;
        else if (argument == "--sigma") {
            options.sigma = parsePositiveNumber(argument, value);
        } else if (argument == "--roughness") {
            options.roughness = parsePositiveNumber(argument, value);
        } else if (argument == "--output") options.outputDirectory = value;
        else if (argument == "--spp") {
            options.samplesPerPixel = parsePositiveInteger(argument, value);
        } else if (argument == "--flight-cells") {
            options.flightTableCells = parsePositiveInteger(argument, value);
            if (*options.flightTableCells < 4) {
                throw std::invalid_argument("--flight-cells must be at least 4");
            }
        } else if (argument == "--width") {
            options.width = parsePositiveInteger(argument, value);
        } else if (argument == "--height") {
            options.height = parsePositiveInteger(argument, value);
        } else if (argument == "--threads") {
            options.threadCount = parseNonnegativeInteger(argument, value);
        } else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.configPath.empty()) throw std::invalid_argument("missing --config <path>");
    if (options.preserveSlope && !options.sigma) {
        throw std::invalid_argument("--preserve-slope requires --sigma");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: macrofacet_experiments {curves|gp-reference|render|all} "
                     "--config <file> [--sigma <value>] [--roughness <value>] [--preserve-slope] "
                     "[--width <pixels>] [--height <pixels>] [--spp <count>] "
                     "[--flight-cells <count>] [--threads <count>] [--output <directory>]\n";
        return 2;
    }
    try {
        const CommandLine options = parseCommandLine(argc, argv);
        const std::string& command = options.command;
        mf::ExperimentConfig config = mf::loadExperimentConfig(options.configPath);
        mf::applyFieldOverrides(config, options.sigma, options.roughness, options.preserveSlope);
        if (options.outputDirectory) config.outputDirectory = *options.outputDirectory;
        if (options.samplesPerPixel) config.render.samplesPerPixel = *options.samplesPerPixel;
        if (options.flightTableCells) config.render.flightTableCells = *options.flightTableCells;
        if (options.width) config.render.width = *options.width;
        if (options.height) config.render.height = *options.height;
        if (options.threadCount) config.render.threadCount = *options.threadCount;
        std::filesystem::create_directories(config.outputDirectory);
        mf::writeResolvedConfig(config, config.outputDirectory / "resolved_config.json");
        if (command == "curves" || command == "all") mf::runFlightCurves(config);
        if (command == "gp-reference" || command == "all") mf::runConditionalGPReference(config);
        if (command == "render" || command == "all") mf::runRenderExperiments(config);
        if (command != "curves" && command != "gp-reference" && command != "render" &&
            command != "all") {
            throw std::invalid_argument("unknown command: " + command);
        }
        std::ofstream summary(config.outputDirectory / "run_summary.json");
        summary << "{\n  \"success\": true,\n  \"command\": \"" << command
                << "\",\n  \"seed\": " << config.seed
                << ",\n  \"sigma\": " << config.field.kernel.sigma() << "\n}\n";
        std::cout << "completed " << command << " -> " << config.outputDirectory.string() << '\n';
        return 0;
    } catch (const mf::NumericError& error) {
        std::cerr << "numeric failure [" << mf::toString(error.status()) << "]: "
                  << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "failure: " << error.what() << '\n';
        return 1;
    }
}

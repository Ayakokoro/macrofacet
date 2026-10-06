#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/RenderExperiment.h"
#include "macrofacet/experiments/TransmittanceCurves.h"
#if defined(MACROFACET_HAS_FIELDS)
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#endif
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
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> threadCount;
    std::optional<std::string> mode;
    std::optional<int> rays;
    std::optional<int> bins;
    std::optional<int> trials;
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
    if (options.command != "render" && options.command != "curves" &&
        options.command != "first-passage")
        throw std::invalid_argument("unknown command: " + options.command);
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
        } else if (argument == "--width") {
            options.width = parsePositiveInteger(argument, value);
        } else if (argument == "--height") {
            options.height = parsePositiveInteger(argument, value);
        } else if (argument == "--threads") {
            options.threadCount = parseNonnegativeInteger(argument, value);
        } else if (argument == "--mode") {
            options.mode = value;
        } else if (argument == "--rays") {
            options.rays = parsePositiveInteger(argument, value);
        } else if (argument == "--bins") {
            options.bins = parsePositiveInteger(argument, value);
        } else if (argument == "--trials") {
            options.trials = parsePositiveInteger(argument, value);
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
#if defined(MACROFACET_HAS_FIELDS)
    // Makes `"mean_type": "nanovdb"` resolvable for every command below. An
    // explicit call, not a static initializer: nothing in this binary references
    // macrofacet_field's translation units, so the linker would drop a static
    // registration object. Idempotent, and cheap enough to do unconditionally.
    mf::registerNanoVdbFieldTypes();
#endif
    if (argc < 2) {
        std::cerr << "usage: macrofacet_experiments render "
                     "--config <file> [--sigma <value>] [--roughness <value>] [--preserve-slope] "
                     "[--width <pixels>] [--height <pixels>] [--spp <count>] "
                     "[--threads <count>] [--output <directory>]\n"
                     "       macrofacet_experiments curves --config <file> "
                     "[--trials <count>] [--bins <count>] [--output <directory>]\n"
                     "       macrofacet_experiments first-passage --config <file> "
                     "[--trials <count>] [--bins <count>] [--threads <count>] "
                     "[--output <directory>]\n";
        return 2;
    }
    try {
        const CommandLine options = parseCommandLine(argc, argv);
        const std::string& command = options.command;
        if (command == "first-passage") {
            if (options.sigma || options.roughness || options.preserveSlope ||
                options.samplesPerPixel || options.width || options.height ||
                options.mode || options.rays) {
                throw std::invalid_argument(
                    "first-passage accepts only --config, --trials, --bins, --threads, and --output");
            }
            mf::FirstPassageExperimentConfig firstPassage =
                mf::loadFirstPassageExperimentConfig(options.configPath);
            if (options.outputDirectory) firstPassage.outputDirectory = *options.outputDirectory;
            if (options.trials) {
                firstPassage.trajectories = *options.trials;
                if (firstPassage.fixedEndpoint.trajectories > 0)
                    firstPassage.fixedEndpoint.trajectories = *options.trials;
            }
            if (options.bins) firstPassage.curveBins = *options.bins;
            if (options.threadCount.has_value()) firstPassage.threadCount = *options.threadCount;
            mf::runFirstPassageExperiment(firstPassage);
            std::ofstream summary(firstPassage.outputDirectory / "run_summary.json");
            summary << "{\n  \"success\": true,\n  \"command\": \"first-passage\",\n"
                    << "  \"seed\": " << firstPassage.seed << ",\n"
                    << "  \"initial_condition_type\": \""
                    << firstPassage.initialConditionType << "\",\n"
                    << "  \"trajectories_per_state_and_resolution\": "
                    << firstPassage.trajectories << ",\n"
                    << "  \"fixed_endpoint_enabled\": " << (firstPassage.fixedEndpoint.enabled ? "true" : "false") << ",\n"
                    << "  \"fixed_endpoint_only\": " << (firstPassage.fixedEndpoint.only ? "true" : "false") << ",\n"
                    << "  \"fixed_endpoint_proposals_per_target_and_resolution\": "
                    << (firstPassage.fixedEndpoint.enabled ? (firstPassage.fixedEndpoint.trajectories > 0
                        ? firstPassage.fixedEndpoint.trajectories : firstPassage.trajectories) : 0) << ",\n"
                    << "  \"state_count\": "
                    << (firstPassage.initialConditionType == "collision_state"
                            ? firstPassage.collisionStates.size() : 1) << ",\n"
                    << "  \"kernel_count\": " << firstPassage.kernels.size() << ",\n"
                    << "  \"resolution_count\": " << firstPassage.stepSizes.size()
                    << "\n}\n";
            std::cout << "completed first-passage -> "
                      << firstPassage.outputDirectory.string() << '\n';
            return 0;
        }
        mf::ExperimentConfig config = mf::loadExperimentConfig(options.configPath);
        mf::applyFieldOverrides(config, options.sigma, options.roughness, options.preserveSlope);
        if (options.outputDirectory) config.outputDirectory = *options.outputDirectory;
        if (options.samplesPerPixel) config.render.samplesPerPixel = *options.samplesPerPixel;
        if (options.width) config.render.width = *options.width;
        if (options.height) config.render.height = *options.height;
        if (options.threadCount) config.render.threadCount = *options.threadCount;
        if (options.mode) config.transportMode = *options.mode;
        if (config.transportMode != "classic" && config.transportMode != "classic_local" &&
            config.transportMode != "classic_global" &&
            config.transportMode != "global_conditional" && config.transportMode != "all")
            throw std::invalid_argument("unknown transport mode: " + config.transportMode);
        if (command == "curves" && config.transmittance) {
            if (options.rays) {
                throw std::invalid_argument(
                    "--rays is only available for legacy configs without transmittance.rays");
            }
            if (options.bins) {
                config.transmittance->bins = *options.bins;
                for (mf::TransmittanceRayConfig& ray : config.transmittance->rays)
                    ray.bins = *options.bins;
            }
            if (options.trials) config.transmittance->trialsPerRay = *options.trials;
        } else if (options.trials) {
            throw std::invalid_argument(
                "--trials requires an explicit transmittance block in the config");
        }
#if defined(MACROFACET_HAS_FIELDS)
        mf::prepareNanoVdbField(config);
#else
        throw std::runtime_error("tracing requires MACROFACET_BUILD_FIELDS=ON and a NanoVDB field");
#endif
        std::filesystem::create_directories(config.outputDirectory);
        mf::writeResolvedConfig(config, config.outputDirectory / "resolved_config.json");
        if (command == "render") mf::runRenderExperiments(config);
        else mf::runTransmittanceCurves(config, options.rays.value_or(1024),
                                        options.bins.value_or(64));
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

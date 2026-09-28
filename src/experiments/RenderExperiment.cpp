#include "macrofacet/experiments/RenderExperiment.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <algorithm>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

class RenderProgressBar {
public:
    explicit RenderProgressBar(std::uint64_t total) : total_(total), reporter_([this] {
        std::uint64_t last = std::numeric_limits<std::uint64_t>::max();
        do {
            const std::uint64_t count = completed.load(std::memory_order_relaxed);
            if (count != last) {
                print(count);
                last = count;
            }
            if (finished_.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (true);
        const std::uint64_t count = completed.load(std::memory_order_relaxed);
        if (count != last) print(count);
        std::cerr << '\n';
    }) {}

    ~RenderProgressBar() {
        finished_.store(true, std::memory_order_release);
        reporter_.join();
    }

    std::atomic<std::uint64_t> completed{0};

private:
    void print(std::uint64_t count) const {
        constexpr int width = 30;
        const double fraction = std::min(1.0, static_cast<double>(count) / total_);
        const int filled = static_cast<int>(width * fraction);
        std::ostringstream line;
        line << "\rrender [" << std::string(filled, '#')
             << std::string(width - filled, '-') << "] "
             << std::setw(3) << static_cast<int>(100.0 * fraction) << "% "
             << count << '/' << total_ << " camera rays";
        std::cerr << line.str() << std::flush;
    }

    const std::uint64_t total_;
    std::atomic<bool> finished_{false};
    std::thread reporter_;
};

std::uint64_t totalCameraRays(const mf::ExperimentConfig& config,
                              std::size_t modeCount, std::size_t environmentCount) {
    std::uint64_t total = 1;
    for (const std::uint64_t factor : {
             static_cast<std::uint64_t>(config.render.width),
             static_cast<std::uint64_t>(config.render.height),
             static_cast<std::uint64_t>(config.render.samplesPerPixel),
             static_cast<std::uint64_t>(modeCount),
             static_cast<std::uint64_t>(environmentCount)}) {
        if (factor == 0 || total > std::numeric_limits<std::uint64_t>::max() / factor)
            throw std::overflow_error("render camera-ray count exceeds uint64 range");
        total *= factor;
    }
    return total;
}

} // namespace

namespace mf {

void runRenderExperiments(const ExperimentConfig& config) {
    requireNanoVdbField(config);
    std::filesystem::create_directories(config.outputDirectory);
    std::ofstream summary(config.outputDirectory / "render_summary.csv");
    summary << "mode,environment,width,height,spp,seconds,paths,real_collisions,escaped_paths,"
               "roulette_terminations,safety_cap_terminations,numerical_failures,mean_path_depth,"
               "dda_candidates,null_collisions,bound_intervals,near_candidates,far_candidates,rounded_candidate_steps,adaptive_majorant_flights\n";
    {
        const std::vector<std::string> modes = config.transportMode == "all"
            ? std::vector<std::string>{"classic_local", "classic_global", "global_conditional"}
            : std::vector<std::string>{config.transportMode};
        std::vector<std::string> environments{"unit_white"};
        if (config.render.environment != "unit_white") environments.push_back(config.render.environment);
        RenderProgressBar progress(totalCameraRays(config, modes.size(), environments.size()));
        for (const std::string& mode : modes) for (const std::string& environment : environments) {
            ExperimentConfig renderConfig = config;
            renderConfig.transportMode = mode;
            if (mode == "classic_local") renderConfig.material.gpModel = GpModel::LocalTangent;
            if (mode == "classic_global" || mode == "global_conditional")
                renderConfig.material.gpModel = GpModel::GlobalPointwise;
            if (renderConfig.material.gpModel != config.material.gpModel)
                renderConfig.preparedAreaMajorant.reset();
            if (mode == "global_conditional" &&
                (renderConfig.material.ndfFamily != NdfFamily::GeneralizedGaussian ||
                 renderConfig.material.alphaField))
                throw std::invalid_argument(
                    "global_conditional requires generalized_gaussian without alpha grid");
            renderConfig.material.validate(renderConfig.field.activeDomain);
            renderConfig.render.environment = environment;
            if (environment == "unit_white") {
                renderConfig.material.conductor.forceUnitFresnel = true;
            }
            const auto start = std::chrono::steady_clock::now();
            const RenderedImage image = renderAnalyticScene(renderConfig, &progress.completed);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const std::string suffix = environment == "unit_white" ? "white" : "directional";
            const std::string stem = "render_" +
                (mode == "classic" ? std::string("classic") : mode) + "_" + suffix;
            writePfm(config.outputDirectory / (stem + ".pfm"), image);
            writeBmpPreview(config.outputDirectory / (stem + ".bmp"), image);
            const double meanDepth = image.statistics.paths > 0
                ? image.statistics.accumulatedPathDepth / image.statistics.paths : 0.0;
            summary << mode << ',' << environment << ',' << image.width << ','
                    << image.height << ',' << config.render.samplesPerPixel << ','
                    << std::setprecision(17) << seconds << ',' << image.statistics.paths << ','
                    << image.statistics.realCollisions << ',' << image.statistics.escapedPaths << ','
                    << image.statistics.rouletteTerminations << ','
                    << image.statistics.safetyCapTerminations << ','
                    << image.statistics.numericalFailures << ',' << meanDepth << ','
                    << image.statistics.tracking.candidates << ','
                    << image.statistics.tracking.nullCollisions << ','
                    << image.statistics.tracking.boundIntervals << ','
                    << image.statistics.tracking.nearCandidates << ','
                    << image.statistics.tracking.farCandidates << ','
                    << image.statistics.tracking.roundedCandidateSteps << ','
                    << image.statistics.tracking.adaptiveMajorantFlights << '\n';
        }
    }
}

} // namespace mf

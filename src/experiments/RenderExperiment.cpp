#include "macrofacet/experiments/RenderExperiment.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace mf {

void runRenderExperiments(const ExperimentConfig& config) {
    std::filesystem::create_directories(config.outputDirectory);
    std::ofstream summary(config.outputDirectory / "render_summary.csv");
    summary << "mode,environment,width,height,spp,seconds,paths,real_collisions,escaped_paths,"
               "roulette_terminations,safety_cap_terminations,numerical_failures,mean_path_depth\n";
    for (ModelMode mode : config.modes) {
        std::vector<std::string> environments{"unit_white"};
        if (config.render.environment != "unit_white") environments.push_back(config.render.environment);
        for (const std::string& environment : environments) {
            ExperimentConfig renderConfig = config;
            renderConfig.render.environment = environment;
            if (environment == "unit_white") {
                renderConfig.field.conductor.forceUnitFresnel = true;
            }
            const auto start = std::chrono::steady_clock::now();
            const RenderedImage image = renderAnalyticScene(mode, renderConfig);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const std::string suffix = environment == "unit_white" ? "white" : "directional";
            const std::string stem = "render_" + modelModeName(mode) + "_" + suffix;
            writePfm(config.outputDirectory / (stem + ".pfm"), image);
            writeBmpPreview(config.outputDirectory / (stem + ".bmp"), image);
            const double meanDepth = image.statistics.paths > 0
                ? image.statistics.accumulatedPathDepth / image.statistics.paths : 0.0;
            summary << modelModeName(mode) << ',' << environment << ',' << image.width << ','
                    << image.height << ',' << config.render.samplesPerPixel << ','
                    << std::setprecision(17) << seconds << ',' << image.statistics.paths << ','
                    << image.statistics.realCollisions << ',' << image.statistics.escapedPaths << ','
                    << image.statistics.rouletteTerminations << ','
                    << image.statistics.safetyCapTerminations << ','
                    << image.statistics.numericalFailures << ',' << meanDepth << '\n';
        }
    }
}

} // namespace mf

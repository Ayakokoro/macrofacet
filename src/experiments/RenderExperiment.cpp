#include "macrofacet/experiments/RenderExperiment.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace mf {

void runRenderExperiments(const ExperimentConfig& config) {
    requireNanoVdbField(config);
    std::filesystem::create_directories(config.outputDirectory);
    std::ofstream summary(config.outputDirectory / "render_summary.csv");
    summary << "mode,environment,width,height,spp,seconds,paths,real_collisions,escaped_paths,"
               "roulette_terminations,safety_cap_terminations,numerical_failures,mean_path_depth,"
               "hazard_evaluations,quadrature_intervals,newton_iterations,bisection_steps,"
               "maximum_optical_residual,maximum_integration_error\n";
    {
        std::vector<std::string> environments{"unit_white"};
        if (config.render.environment != "unit_white") environments.push_back(config.render.environment);
        for (const std::string& environment : environments) {
            ExperimentConfig renderConfig = config;
            renderConfig.render.environment = environment;
            if (environment == "unit_white") {
                renderConfig.material.conductor.forceUnitFresnel = true;
            }
            const auto start = std::chrono::steady_clock::now();
            const RenderedImage image = renderAnalyticScene(renderConfig);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const std::string suffix = environment == "unit_white" ? "white" : "directional";
            const std::string stem = "render_classic_" + suffix;
            writePfm(config.outputDirectory / (stem + ".pfm"), image);
            writeBmpPreview(config.outputDirectory / (stem + ".bmp"), image);
            const double meanDepth = image.statistics.paths > 0
                ? image.statistics.accumulatedPathDepth / image.statistics.paths : 0.0;
            summary << "classic" << ',' << environment << ',' << image.width << ','
                    << image.height << ',' << config.render.samplesPerPixel << ','
                    << std::setprecision(17) << seconds << ',' << image.statistics.paths << ','
                    << image.statistics.realCollisions << ',' << image.statistics.escapedPaths << ','
                    << image.statistics.rouletteTerminations << ','
                    << image.statistics.safetyCapTerminations << ','
                    << image.statistics.numericalFailures << ',' << meanDepth << ','
                    << image.statistics.tracking.hazardEvaluations << ','
                    << image.statistics.tracking.quadratureIntervals << ','
                    << image.statistics.tracking.newtonIterations << ','
                    << image.statistics.tracking.bisectionSteps << ','
                    << image.statistics.tracking.maximumResidual << ','
                    << image.statistics.tracking.maximumIntegrationError << '\n';
        }
    }
}

} // namespace mf

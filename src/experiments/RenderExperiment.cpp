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
               "dda_candidates,null_collisions,bound_intervals,near_candidates,far_candidates,rounded_candidate_steps,adaptive_majorant_flights\n";
    {
        const std::vector<std::string> modes = config.transportMode == "all"
            ? std::vector<std::string>{"classic_local", "classic_global", "global_conditional"}
            : std::vector<std::string>{config.transportMode};
        std::vector<std::string> environments{"unit_white"};
        if (config.render.environment != "unit_white") environments.push_back(config.render.environment);
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
            const RenderedImage image = renderAnalyticScene(renderConfig);
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

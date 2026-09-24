#include "macrofacet/experiments/FlightCurveExperiment.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

namespace mf {

void runFlightCurves(const ExperimentConfig& config) {
    std::filesystem::create_directories(config.outputDirectory);
    const FlightState state = startSurfaceFlight(config.fixedFlight.birthPosition,
                                                 config.fixedFlight.birthGradient,
                                                 config.fixedFlight.direction);
    std::ofstream curves(config.outputDirectory / "flight_curves.csv");
    curves << "mode,t,h,log_h,H,T_model,p_model,log_screen_probability,log_crossing_flux,"
              "integration_error,numeric_status,elapsed_microseconds\n";
    std::ofstream histograms(config.outputDirectory / "flight_histograms.csv");
    histograms << "mode,bin_left,bin_right,observed_mass,expected_mass,escape_bin\n";

    for (ModelMode mode : config.modes) {
        std::unique_ptr<FlightKernel> kernel = makeFlightKernel(
            mode, config.field, state, config.conditional29.externalPolicy, config.numeric);
        const double maximumAge = std::min(config.fixedFlight.requestedMaximumAge,
                                           kernel->maximumAgeInDomain());
        const int count = config.fixedFlight.curveSampleCount;
        std::vector<double> ages(static_cast<std::size_t>(count));
        std::vector<double> opticalDepth(static_cast<std::size_t>(count), 0.0);
        std::vector<double> hazard(static_cast<std::size_t>(count), 0.0);
        for (int i = 0; i < count; ++i) ages[static_cast<std::size_t>(i)] =
            maximumAge * i / (count - 1.0);
        for (int i = 0; i < count; ++i) {
            const auto started = std::chrono::steady_clock::now();
            const HazardEvaluation evaluation = kernel->evaluate(ages[static_cast<std::size_t>(i)]);
            hazard[static_cast<std::size_t>(i)] = evaluation.hazard.value;
            PositiveResult integrated = exactZero();
            if (i > 0) {
                integrated = integrateHazard(*kernel, ages[static_cast<std::size_t>(i - 1)],
                                             ages[static_cast<std::size_t>(i)], config.numeric);
                opticalDepth[static_cast<std::size_t>(i)] =
                    opticalDepth[static_cast<std::size_t>(i - 1)] +
                    (integrated.status == NumericStatus::ExactZero ? 0.0 : integrated.value);
            }
            const double survival = std::exp(-opticalDepth[static_cast<std::size_t>(i)]);
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
            curves << modelModeName(mode) << ',' << std::setprecision(17)
                   << ages[static_cast<std::size_t>(i)] << ',' << evaluation.hazard.value << ','
                   << evaluation.hazard.logValue << ',' << opticalDepth[static_cast<std::size_t>(i)]
                   << ',' << survival << ',' << evaluation.hazard.value * survival << ',';
            if (evaluation.logExteriorScreenProbability) curves << *evaluation.logExteriorScreenProbability;
            curves << ',';
            if (evaluation.logCrossingFlux) curves << *evaluation.logCrossingFlux;
            curves << ',' << integrated.absError << ',' << toString(evaluation.hazard.status) << ','
                   << elapsed << '\n';
        }

        std::vector<int> bins(static_cast<std::size_t>(count - 1), 0);
        int escapes = 0;
        Random rng(config.seed + static_cast<unsigned>(mode) * 7919ULL);
        std::unique_ptr<OpticalDepthSampler> regular;
        if (mode == ModelMode::Conditional29)
            regular = std::make_unique<OpticalDepthSampler>(*kernel, config.numeric, maximumAge);
        for (int sample = 0; sample < config.fixedFlight.flightSampleCount; ++sample) {
            if (regular) {
                const auto flight = regular->sample(rng);
                if (!flight.collided) { ++escapes; continue; }
                const auto upper = std::upper_bound(ages.begin(), ages.end(), flight.age);
                const int index = std::clamp(static_cast<int>(upper - ages.begin()) - 1, 0, count - 2);
                ++bins[static_cast<std::size_t>(index)];
                continue;
            }
            const double target = -std::log1p(-rng.openUniform01());
            if (target >= opticalDepth.back()) { ++escapes; continue; }
            const auto upper = std::upper_bound(opticalDepth.begin(), opticalDepth.end(), target);
            const int index = std::clamp(static_cast<int>(upper - opticalDepth.begin()) - 1,
                                         0, count - 2);
            ++bins[static_cast<std::size_t>(index)];
        }
        for (int i = 0; i < count - 1; ++i) {
            const double expected = std::exp(-opticalDepth[static_cast<std::size_t>(i)]) -
                                    std::exp(-opticalDepth[static_cast<std::size_t>(i + 1)]);
            histograms << modelModeName(mode) << ',' << std::setprecision(17)
                       << ages[static_cast<std::size_t>(i)] << ','
                       << ages[static_cast<std::size_t>(i + 1)] << ','
                       << static_cast<double>(bins[static_cast<std::size_t>(i)]) /
                              config.fixedFlight.flightSampleCount
                       << ',' << expected << ",0\n";
        }
        histograms << modelModeName(mode) << ',' << maximumAge << ',' << maximumAge << ','
                   << static_cast<double>(escapes) / config.fixedFlight.flightSampleCount << ','
                   << std::exp(-opticalDepth.back()) << ",1\n";
    }
}

} // namespace mf

#include "macrofacet/experiments/ConditionalGPReference.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mf {

ReferenceResult sampleConditionalGPFirstHit(const ConditionedRay& ray,
                                            const std::vector<double>& positiveAges,
                                            int sampleCount, std::uint64_t seed) {
    if (sampleCount <= 0 || positiveAges.empty()) throw std::invalid_argument("invalid GP reference budget");
    const DynamicGaussian path = ray.valuesAt(positiveAges);
    const FactoredGaussianPSD pathFactor = factorGaussianPSD(path);
    std::vector<int> survivalCounts(positiveAges.size(), 0);
    std::vector<int> hitCounts(positiveAges.size(), 0);
    Random rng(seed);
    for (int sample = 0; sample < sampleCount; ++sample) {
        const Eigen::VectorXd values = sampleGaussianPSD(pathFactor, rng);
        bool alive = true;
        for (std::size_t i = 0; i < positiveAges.size(); ++i) {
            if (alive && values[static_cast<int>(i)] > 0.0) {
                ++survivalCounts[i];
            } else if (alive) {
                ++hitCounts[i];
                alive = false;
            }
        }
    }
    ReferenceResult result;
    result.ages = positiveAges;
    result.sampleCount = sampleCount;
    result.survival.resize(positiveAges.size());
    result.firstHitMass.resize(positiveAges.size());
    result.averageHazard.resize(positiveAges.size());
    result.standardError.resize(positiveAges.size());
    double previousSurvival = 1.0;
    double previousAge = 0.0;
    for (std::size_t i = 0; i < positiveAges.size(); ++i) {
        const double survival = static_cast<double>(survivalCounts[i]) / sampleCount;
        result.survival[i] = survival;
        result.firstHitMass[i] = static_cast<double>(hitCounts[i]) / sampleCount;
        result.standardError[i] = std::sqrt(survival * (1.0 - survival) / sampleCount);
        if (survival > 0.0 && previousSurvival > 0.0) {
            result.averageHazard[i] = -std::log(survival / previousSurvival) /
                                      (positiveAges[i] - previousAge);
        } else {
            result.averageHazard[i] = std::numeric_limits<double>::quiet_NaN();
        }
        previousSurvival = survival;
        previousAge = positiveAges[i];
    }
    return result;
}

ScreenedFormulaEstimate estimateScreenedFirstPassageHazard(
    const ConditionedRay& ray, double t, const std::vector<double>& interiorAges,
    int sampleCount, std::uint64_t seed) {
    if (!(t > 0.0) || sampleCount <= 0) throw std::invalid_argument("invalid screened formula request");
    const DynamicGaussian joint = ray.checkpointValuesAndEndpointSlope(interiorAges, t);
    const int n = static_cast<int>(interiorAges.size());

    DynamicGaussian exterior;
    exterior.mean = joint.mean.head(n + 1);
    exterior.covariance = joint.covariance.topLeftCorner(n + 1, n + 1);
    const FactoredGaussianPSD exteriorFactor = factorGaussianPSD(exterior);
    Random denominatorRng(seed);
    int survivors = 0;
    for (int sample = 0; sample < sampleCount; ++sample) {
        const Eigen::VectorXd values = sampleGaussianPSD(exteriorFactor, denominatorRng);
        if ((values.array() > 0.0).all()) ++survivors;
    }
    const double exteriorProbability = static_cast<double>(survivors) / sampleCount;

    DynamicGaussian target;
    target.mean = Eigen::VectorXd(n + 1);
    target.covariance = Eigen::MatrixXd(n + 1, n + 1);
    for (int i = 0; i < n; ++i) target.mean[i] = joint.mean[i];
    target.mean[n] = joint.mean[n + 1];
    for (int i = 0; i <= n; ++i) {
        const int sourceI = i < n ? i : n + 1;
        for (int j = 0; j <= n; ++j) {
            const int sourceJ = j < n ? j : n + 1;
            target.covariance(i, j) = joint.covariance(sourceI, sourceJ);
        }
    }
    DynamicGaussian endpoint;
    endpoint.mean = Eigen::VectorXd::Constant(1, joint.mean[n]);
    endpoint.covariance = Eigen::MatrixXd::Constant(1, 1, joint.covariance(n, n));
    Eigen::MatrixXd cross(n + 1, 1);
    for (int i = 0; i <= n; ++i) {
        const int source = i < n ? i : n + 1;
        cross(i, 0) = joint.covariance(source, n);
    }
    const DynamicGaussian conditional = conditionGaussianDynamic(
        target, endpoint, cross, Eigen::VectorXd::Zero(1));
    const FactoredGaussianPSD conditionalFactor = factorGaussianPSD(conditional);
    Random numeratorRng(seed ^ 0x9e3779b97f4a7c15ULL);
    double sum = 0.0;
    double sumSquares = 0.0;
    for (int sample = 0; sample < sampleCount; ++sample) {
        const Eigen::VectorXd values = sampleGaussianPSD(conditionalFactor, numeratorRng);
        bool screen = true;
        for (int i = 0; i < n; ++i) screen = screen && values[i] > 0.0;
        const double weight = screen ? std::max(-values[n], 0.0) : 0.0;
        sum += weight;
        sumSquares += weight * weight;
    }
    const double meanWeight = sum / sampleCount;
    const double varianceWeight = std::max(0.0, sumSquares / sampleCount - meanWeight * meanWeight);
    const double stddevF = std::sqrt(endpoint.covariance(0, 0));
    const double endpointDensity = std::exp(normalLogPdf(-endpoint.mean[0] / stddevF)) / stddevF;
    const double crossingFlux = endpointDensity * meanWeight;
    const double exteriorError = std::sqrt(exteriorProbability * (1.0 - exteriorProbability) /
                                           sampleCount);
    const double fluxError = endpointDensity * std::sqrt(varianceWeight / sampleCount);

    ScreenedFormulaEstimate result;
    result.exteriorProbability = exteriorProbability;
    result.crossingFlux = crossingFlux;
    result.exteriorStandardError = exteriorError;
    result.fluxStandardError = fluxError;
    result.sampleCount = sampleCount;
    if (survivors == 0) {
        result.status = NumericStatus::UnresolvedRareEvent;
        result.hazard = std::numeric_limits<double>::quiet_NaN();
        result.hazardStandardError = std::numeric_limits<double>::quiet_NaN();
    } else {
        result.hazard = crossingFlux / exteriorProbability;
        const double relativeFlux = crossingFlux > 0.0 ? fluxError / crossingFlux : 0.0;
        const double relativeExterior = exteriorError / exteriorProbability;
        result.hazardStandardError = result.hazard *
            std::sqrt(relativeFlux * relativeFlux + relativeExterior * relativeExterior);
    }
    return result;
}

void runConditionalGPReference(const ExperimentConfig& config) {
    std::filesystem::create_directories(config.outputDirectory);
    const FlightState state = startSurfaceFlight(config.fixedFlight.birthPosition,
                                                 config.fixedFlight.birthGradient,
                                                 config.fixedFlight.direction);
    ConditionedRay ray(config.field, state.birthPosition, state.birthGradient, state.direction,
                       config.numeric);
    const DomainInterval domain = config.field.activeDomain.intersect({state.birthPosition,
                                                                       state.direction});
    const double maximumAge = std::min(config.fixedFlight.requestedMaximumAge, domain.exit);
    std::ofstream pathStream(config.outputDirectory / "gp_reference_paths.csv");
    pathStream << "grid_intervals,age,survival,standard_error,first_hit_mass,average_hazard\n";
    for (int intervals : config.reference.nestedGridIntervals) {
        if (intervals > config.reference.maxGridPoints) continue;
        std::vector<double> ages(static_cast<std::size_t>(intervals));
        for (int i = 0; i < intervals; ++i) ages[static_cast<std::size_t>(i)] =
            maximumAge * (i + 1.0) / intervals;
        const ReferenceResult result = sampleConditionalGPFirstHit(
            ray, ages, config.reference.pathSampleCount, config.seed + intervals);
        for (std::size_t i = 0; i < ages.size(); ++i) {
            pathStream << intervals << ',' << std::setprecision(17) << ages[i] << ','
                       << result.survival[i] << ',' << result.standardError[i] << ','
                       << result.firstHitMass[i] << ',' << result.averageHazard[i] << '\n';
        }
    }

    std::ofstream formulaStream(config.outputDirectory / "screened_formula.csv");
    formulaStream << "age,checkpoint_count,U_N,U_standard_error,J_N,J_standard_error,h_N,h_standard_error,status\n";
    for (int ai = 1; ai <= config.reference.formulaAgeCount; ++ai) {
        const double age = maximumAge * ai / config.reference.formulaAgeCount;
        for (int count : config.reference.formulaCheckpointCounts) {
            std::vector<double> checkpoints(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) checkpoints[static_cast<std::size_t>(i)] =
                age * (i + 1.0) / (count + 1.0);
            const auto estimate = estimateScreenedFirstPassageHazard(
                ray, age, checkpoints, config.reference.formulaSampleCount,
                config.seed + 1000003ULL * ai + count);
            formulaStream << std::setprecision(17) << age << ',' << count << ','
                          << estimate.exteriorProbability << ',' << estimate.exteriorStandardError << ','
                          << estimate.crossingFlux << ',' << estimate.fluxStandardError << ','
                          << estimate.hazard << ',' << estimate.hazardStandardError << ','
                          << toString(estimate.status) << '\n';
        }
    }
}

} // namespace mf

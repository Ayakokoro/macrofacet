#pragma once

#include "macrofacet/gpss/ConditionedRay.h"
#include <cstdint>
#include <vector>

namespace mf {

struct ReferenceResult {
    std::vector<double> ages;
    std::vector<double> survival;
    std::vector<double> firstHitMass;
    std::vector<double> averageHazard;
    std::vector<double> standardError;
    int sampleCount = 0;
};

struct ScreenedFormulaEstimate {
    double exteriorProbability = 0.0;
    double crossingFlux = 0.0;
    double hazard = 0.0;
    double exteriorStandardError = 0.0;
    double fluxStandardError = 0.0;
    double hazardStandardError = 0.0;
    NumericStatus status = NumericStatus::Ok;
    int sampleCount = 0;
};

ReferenceResult sampleConditionalGPFirstHit(const ConditionedRay& ray,
                                            const std::vector<double>& positiveAges,
                                            int sampleCount, std::uint64_t seed);
ScreenedFormulaEstimate estimateScreenedFirstPassageHazard(
    const ConditionedRay& ray, double t, const std::vector<double>& interiorAges,
    int sampleCount, std::uint64_t seed);

struct ExperimentConfig;
void runConditionalGPReference(const ExperimentConfig& config);

} // namespace mf

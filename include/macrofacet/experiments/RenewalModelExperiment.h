#pragma once
#include "macrofacet/experiments/FirstPassageExperiment.h"

namespace mf {
// Reuses reference ray/mean/start configurations; performs learned-model queries.
void runRenewalModelExperiment(const FirstPassageExperimentConfig& config,
    const std::filesystem::path& bundle, const std::filesystem::path& source);
} // namespace mf

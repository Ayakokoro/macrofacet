#pragma once

#include "macrofacet/experiments/ExperimentConfig.h"

namespace mf {

// Resolve an experiment's procedural mean to a cached, full-domain NanoVDB
// before any tracing command runs. Existing NanoVDB inputs are left as-is.
void prepareNanoVdbField(ExperimentConfig& config);

} // namespace mf

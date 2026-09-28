#pragma once

namespace mf {
struct ExperimentConfig;
void runTransmittanceCurves(const ExperimentConfig& config, int rayCount, int bins);
} // namespace mf

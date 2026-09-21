#pragma once

#include "macrofacet/transport/FlightKernel.h"

namespace mf {

class ClassicFlightKernel final : public FlightKernel {
public:
    ClassicFlightKernel(const GPSSField& field, const FlightState& state)
        : FlightKernel(field, state) {}
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    ModelMode mode() const override { return ModelMode::Classic; }
};

} // namespace mf


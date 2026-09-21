#pragma once

#include "macrofacet/transport/FlightKernel.h"

namespace mf {

class MidpointFlightKernel final : public FlightKernel {
public:
    MidpointFlightKernel(const GPSSField& field, const FlightState& state, bool enableScreen);
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    ModelMode mode() const override { return ModelMode::Midpoint; }
    const ConditionedRay& conditionedRay() const { return ray_; }
private:
    ConditionedRay ray_;
    bool enableScreen_;
};

} // namespace mf


#pragma once

#include "macrofacet/transport/FlightKernel.h"

namespace mf {

class Conditional29FlightKernel final : public FlightKernel {
public:
    Conditional29FlightKernel(const GPSSField& field, const FlightState& state,
                              const NumericPolicy& policy = defaultNumericPolicy());
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    ModelMode mode() const override { return ModelMode::Conditional29; }
    const ConditionedRay& conditionedRay() const { return ray_; }
private:
    ConditionedRay ray_;
    NumericPolicy policy_;
};

} // namespace mf

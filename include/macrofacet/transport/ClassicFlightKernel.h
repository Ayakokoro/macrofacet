#pragma once

#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/macrofacet/MaterialConfig.h"

namespace mf {

class ClassicFlightKernel final : public FlightKernel {
public:
    ClassicFlightKernel(const GPSSField& field, const MaterialConfig& material,
                        const FlightState& state, ScalarFieldPtr density = nullptr)
        : FlightKernel(field, state), material_(material), density_(std::move(density)) {}
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    ModelMode mode() const override { return ModelMode::Classic; }
private:
    MaterialConfig material_;
    ScalarFieldPtr density_;
};

} // namespace mf

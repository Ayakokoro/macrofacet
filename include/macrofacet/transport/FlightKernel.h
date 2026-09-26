#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/macrofacet/MaterialConfig.h"
#include "macrofacet/mathutility/SmallGaussian.h"
#include "macrofacet/transport/FlightState.h"

namespace mf {

struct HazardEvaluation {
    PositiveResult hazard;
};

struct HitStatistics {
    Gaussian<3> collisionGradient;
};

class FlightKernel {
public:
    FlightKernel(const GPSSField& field, FlightState state);
    virtual ~FlightKernel() = default;
    virtual HazardEvaluation evaluate(double age) const = 0;
    virtual HitStatistics hitStatistics(double age) const = 0;
    double currentAge() const { return state_.age; }
    double maximumAgeInDomain() const { return maximumAge_; }
    const FlightState& state() const { return state_; }
    const GPSSField& field() const { return field_; }
protected:
    const GPSSField& field_;
    FlightState state_;
    double maximumAge_ = 0.0;
};

} // namespace mf

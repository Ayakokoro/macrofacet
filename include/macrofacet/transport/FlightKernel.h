#pragma once

#include "macrofacet/gpss/ConditionedRay.h"
#include "macrofacet/transport/FlightState.h"
#include <memory>
#include <optional>

namespace mf {

struct HazardEvaluation {
    PositiveResult hazard;
    std::optional<double> logExteriorScreenProbability;
    std::optional<double> logCrossingFlux;
};

struct HitStatistics {
    Gaussian<3> gradientGivenEndpointZero;
    std::optional<Gaussian<4>> midpointAndGradientGivenEndpointZero; // (Y,Gx,Gy,Gz)
};

class FlightKernel {
public:
    FlightKernel(const GPSSField& field, FlightState state);
    virtual ~FlightKernel() = default;
    virtual HazardEvaluation evaluate(double age) const = 0;
    virtual HitStatistics hitStatistics(double age) const = 0;
    virtual ModelMode mode() const = 0;
    double currentAge() const { return state_.age; }
    double maximumAgeInDomain() const { return maximumAge_; }
    const FlightState& state() const { return state_; }
    const GPSSField& field() const { return field_; }
protected:
    const GPSSField& field_;
    FlightState state_;
    double maximumAge_ = 0.0;
};

std::unique_ptr<FlightKernel> makeFlightKernel(
    ModelMode mode, const GPSSField& field, const FlightState& state,
    ExternalPolicy policy = ExternalPolicy::OriginalMacrofacet,
    const NumericPolicy& numeric = defaultNumericPolicy());

} // namespace mf

#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/transport/MidpointFlightKernel.h"
#include <stdexcept>

namespace mf {

FlightKernel::FlightKernel(const GPSSField& field, FlightState state)
    : field_(field), state_(std::move(state)) {
    const Ray ray{state_.birthPosition, state_.direction};
    const DomainInterval interval = field_.activeDomain.intersect(ray);
    if (!interval.hit || interval.exit < state_.age) {
        throw std::invalid_argument("flight does not intersect the active domain at its current age");
    }
    maximumAge_ = interval.exit;
}

std::unique_ptr<FlightKernel> makeFlightKernel(ModelMode mode, const GPSSField& field,
                                               const FlightState& state, ExternalPolicy policy,
                                               const NumericPolicy& numeric) {
    if (mode == ModelMode::Classic) return std::make_unique<ClassicFlightKernel>(field, state);
    if (field.ndfFamily == NdfFamily::GGXBaseline) {
        throw std::invalid_argument("conditional GP flight is incompatible with a GGX baseline");
    }
    if (state.birthKind == BirthKind::External) {
        if (mode == ModelMode::Conditional29 && policy == ExternalPolicy::SampledExterior) {
            throw std::invalid_argument("sampled_exterior must initialize an observed exterior state");
        }
        return std::make_unique<ClassicFlightKernel>(field, state);
    }
    if (!state.hasFullGradient || (state.birthKind == BirthKind::Surface &&
                                  !(state.direction.dot(state.birthGradient) > 0.0))) {
        throw std::invalid_argument("conditional flight requires an outward full-gradient surface birth");
    }
    if (mode == ModelMode::Conditional29) {
        return std::make_unique<Conditional29FlightKernel>(field, state, numeric);
    }
    if (state.birthKind == BirthKind::ObservedExterior)
        throw std::invalid_argument("observed exterior is currently supported only by conditional29");
    return std::make_unique<MidpointFlightKernel>(field, state, true);
}

} // namespace mf

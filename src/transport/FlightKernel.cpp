#include "macrofacet/transport/FlightKernel.h"
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

} // namespace mf

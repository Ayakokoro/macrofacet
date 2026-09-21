#include "macrofacet/gpss/GPSSField.h"
#include <stdexcept>

namespace mf {

PointPrior GPSSField::pointPrior(const Point3& x) const {
    if (!mean) throw std::invalid_argument("GPSS field has no mean field");
    const MeanJet jet = mean->evaluate(x);
    const double variance = kernel.sigma() * kernel.sigma();
    return {jet.value, variance, jet.gradient, variance * kernel.precision(), Vector3::Zero()};
}

void GPSSField::validate() const {
    if (!mean || !activeDomain.valid()) throw std::invalid_argument("invalid GPSS field");
    if ((conductor.eta.array() < 0.0).any() || (conductor.k.array() < 0.0).any() ||
        !conductor.eta.allFinite() || !conductor.k.allFinite() ||
        ((conductor.eta.array().square() + conductor.k.array().square()) <= 0.0).any()) {
        throw std::invalid_argument("invalid conductor optical constants");
    }
    if (ndfFamily == NdfFamily::GGXBaseline &&
        (!(ggxAlpha.x() > 0.0) || !(ggxAlpha.y() > 0.0))) {
        throw std::invalid_argument("GGX alpha must be positive");
    }
}

} // namespace mf


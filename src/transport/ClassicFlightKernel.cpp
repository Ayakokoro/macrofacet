#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include <stdexcept>

namespace mf {

HazardEvaluation ClassicFlightKernel::evaluate(double age) const {
    if (!(age >= currentAge() && age <= maximumAgeInDomain())) {
        throw std::out_of_range("classic flight age lies outside its domain interval");
    }
    const Point3 point = state_.birthPosition + age * state_.direction;
    return {evaluateClassic(field_, point, state_.direction).extinction, std::nullopt, std::nullopt};
}

HitStatistics ClassicFlightKernel::hitStatistics(double age) const {
    if (field_.ndfFamily == NdfFamily::GGXBaseline) {
        throw NumericError(NumericStatus::UnsupportedDegenerateNdf,
                           "GGX does not define a Gaussian collision gradient");
    }
    const Point3 point = state_.birthPosition + age * state_.direction;
    const PointPrior prior = field_.pointPrior(point);
    Gaussian<3> gradient{prior.meanG, prior.covarianceG};
    return {gradient, std::nullopt};
}

} // namespace mf


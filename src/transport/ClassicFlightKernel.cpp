#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include <cmath>
#include <stdexcept>

namespace mf {

HazardEvaluation ClassicFlightKernel::evaluate(double age) const {
    if (!(age >= currentAge() && age <= maximumAgeInDomain())) {
        throw std::out_of_range("classic flight age lies outside its domain interval");
    }
    const Point3 point = state_.birthPosition + age * state_.direction;
    if (!density_) {
        return {evaluateClassic(field_, material_, point, state_.direction).extinction};
    }
    const double density = density_->sample(point);
    if (!(density >= 0.0) || !std::isfinite(density)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid baked density");
    }
    if (density == 0.0) {
        return {exactZero()};
    }
    const PositiveResult area = classicProjectedArea(field_, material_, point, state_.direction);
    if (area.status == NumericStatus::ExactZero)
        return {exactZero()};
    return {positiveFromLog(std::log(density) + area.logValue)};
}

HitStatistics ClassicFlightKernel::hitStatistics(double age) const {
    if (material_.ndfFamily == NdfFamily::GGXBaseline) {
        throw NumericError(NumericStatus::UnsupportedDegenerateNdf,
                           "GGX does not define a Gaussian collision gradient");
    }
    const Point3 point = state_.birthPosition + age * state_.direction;
    // The collision gradient is the microfacet normal the path reflects off, so
    // it is the material NDF -- the same distribution ConductorPhase uses for
    // the equivalent GGX-family bounce.
    const PointPrior prior = material_.materialNdf(field_, point);
    Gaussian<3> gradient{prior.meanG, prior.covarianceG};
    return {gradient};
}

} // namespace mf

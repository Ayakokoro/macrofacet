#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <cmath>
#include <limits>

namespace mf {

Conditional29FlightKernel::Conditional29FlightKernel(const GPSSField& field,
                                                     const FlightState& state, const NumericPolicy& policy)
    : FlightKernel(field, state),
      ray_(field, state.birthPosition, state.birthValue, state.birthGradient, state.direction, policy),
      policy_(policy) {
    if (!(state.direction.dot(field.kernel.precision() * state.direction) > 0.0)) {
        throw NumericError(NumericStatus::UnsupportedSingularFlight,
                           "conditional29 requires nonzero kernel precision along the ray");
    }
}

HazardEvaluation Conditional29FlightKernel::evaluate(double age) const {
    if (!(age >= currentAge() && age <= maximumAgeInDomain())) {
        throw std::out_of_range("conditional flight age lies outside its domain interval");
    }
    if (age == 0.0) return {exactZero(), 0.0, -std::numeric_limits<double>::infinity()};
    const Gaussian<2> fk = ray_.endpointValueSlope(age);
    const double varianceF = fk.covariance(0, 0);
    if (!(varianceF > 0.0)) {
        throw NumericError(NumericStatus::UnsupportedSingularFlight,
                           "deterministic endpoint zero cannot be represented by a continuous hazard");
    }
    const Gaussian<1> kGivenZero = ray_.slopeGivenEndpointZero(age);
    const double stddevF = std::sqrt(varianceF);
    const double logU = normalLogCdf(fk.mean[0] / stddevF);
    const double varianceK = kGivenZero.covariance(0, 0);
    const PositiveResult moment = negativePartMean(kGivenZero.mean[0], std::sqrt(varianceK), policy_);
    if (moment.status == NumericStatus::ExactZero) {
        return {exactZero(), logU, -std::numeric_limits<double>::infinity()};
    }
    const double logPdfF0 = normalLogPdf(-fk.mean[0] / stddevF) - std::log(stddevF);
    const double logJ = logPdfF0 + moment.logValue;
    const double zeta = fk.mean[0] / stddevF;
    const double logRho = zeta < -32.0
        ? std::log(normalPdfOverCdf(zeta)) - std::log(stddevF) : logPdfF0 - logU;
    return {positiveFromLog(logRho + moment.logValue), logU, logJ};
}

HitStatistics Conditional29FlightKernel::hitStatistics(double age) const {
    if (!(age > 0.0 && age >= currentAge() && age <= maximumAgeInDomain()))
        throw std::out_of_range("invalid conditional collision age");
    return {ray_.gradientGivenEndpointZero(age), std::nullopt};
}

} // namespace mf

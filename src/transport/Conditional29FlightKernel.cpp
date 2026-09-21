#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <cmath>
#include <limits>

namespace mf {

Conditional29FlightKernel::Conditional29FlightKernel(const GPSSField& field,
                                                     const FlightState& state)
    : FlightKernel(field, state),
      ray_(field, state.birthPosition, state.birthGradient, state.direction) {}

HazardEvaluation Conditional29FlightKernel::evaluate(double age) const {
    if (!(age >= currentAge() && age <= maximumAgeInDomain())) {
        throw std::out_of_range("conditional flight age lies outside its domain interval");
    }
    if (age == 0.0) return {exactZero(), 0.0, -std::numeric_limits<double>::infinity()};
    const Gaussian<2> fk = ray_.endpointValueSlope(age);
    const double tolerance = covarianceTolerance(std::max(1.0, fk.covariance.norm()));
    const double varianceF = fk.covariance(0, 0);
    if (varianceF <= tolerance) {
        if (fk.mean[0] > 0.0) return {exactZero(), 0.0, -std::numeric_limits<double>::infinity()};
        throw NumericError(NumericStatus::UnsupportedSingularFlight,
                           "deterministic endpoint zero cannot be represented by a continuous hazard");
    }
    Gaussian<1> target;
    target.mean[0] = fk.mean[1];
    target.covariance(0, 0) = fk.covariance(1, 1);
    Gaussian<1> observation;
    observation.mean[0] = fk.mean[0];
    observation.covariance(0, 0) = varianceF;
    Eigen::Matrix<double, 1, 1> cross;
    cross(0, 0) = fk.covariance(1, 0);
    const Gaussian<1> kGivenZero = conditionGaussian(target, observation, cross,
                                                     Eigen::Matrix<double, 1, 1>::Zero());
    const double stddevF = std::sqrt(varianceF);
    const double logU = normalLogCdf(fk.mean[0] / stddevF);
    const double varianceK = validateNonnegative(kGivenZero.covariance(0, 0),
                                                 fk.covariance.norm());
    const PositiveResult moment = negativePartMean(kGivenZero.mean[0], std::sqrt(varianceK));
    if (moment.status == NumericStatus::ExactZero) {
        return {exactZero(), logU, -std::numeric_limits<double>::infinity()};
    }
    const double logPdfF0 = normalLogPdf(-fk.mean[0] / stddevF) - std::log(stddevF);
    const double logJ = logPdfF0 + moment.logValue;
    return {positiveFromLog(logJ - logU), logU, logJ};
}

HitStatistics Conditional29FlightKernel::hitStatistics(double age) const {
    const Gaussian<4> fg = ray_.endpointValueGradient(age);
    Gaussian<3> target;
    target.mean = fg.mean.template segment<3>(1);
    target.covariance = fg.covariance.template block<3, 3>(1, 1);
    Gaussian<1> observation;
    observation.mean[0] = fg.mean[0];
    observation.covariance(0, 0) = fg.covariance(0, 0);
    Eigen::Matrix<double, 3, 1> cross = fg.covariance.template block<3, 1>(1, 0);
    const Gaussian<3> conditioned = conditionGaussian(target, observation, cross,
                                                      Eigen::Matrix<double, 1, 1>::Zero());
    return {conditioned, std::nullopt};
}

} // namespace mf


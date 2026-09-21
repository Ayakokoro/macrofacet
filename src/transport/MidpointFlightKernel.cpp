#include "macrofacet/transport/MidpointFlightKernel.h"
#include "macrofacet/transport/Conditional29FlightKernel.h"
#include "macrofacet/mathutility/BivariateGaussian.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/mathutility/GaussianScreenIntegral.h"
#include <cmath>
#include <limits>
#include <string>

namespace mf {

MidpointFlightKernel::MidpointFlightKernel(const GPSSField& field, const FlightState& state,
                                           bool enableScreen)
    : FlightKernel(field, state),
      ray_(field, state.birthPosition, state.birthGradient, state.direction),
      enableScreen_(enableScreen) {}

HazardEvaluation MidpointFlightKernel::evaluate(double age) const {
    if (!enableScreen_) {
        Conditional29FlightKernel equivalent(field_, state_);
        return equivalent.evaluate(age);
    }
    if (!(age >= currentAge() && age <= maximumAgeInDomain())) {
        throw std::out_of_range("midpoint flight age lies outside its domain interval");
    }
    if (age == 0.0) return {exactZero(), 0.0, -std::numeric_limits<double>::infinity()};
    const Gaussian<3> yfk = ray_.midpointValueSlope(age);
    const Gaussian<2> yf = marginal<2>(yfk, {0, 1});
    const PositiveResult denominator = positiveOrthant2(yf.mean, yf.covariance);
    if (denominator.status == NumericStatus::ExactZero) {
        throw NumericError(NumericStatus::NeedHigherPrecision,
                           "midpoint exterior-screen probability is exactly zero at age=" +
                           std::to_string(age) + ", means=" + std::to_string(yf.mean[0]) +
                           "," + std::to_string(yf.mean[1]));
    }

    Gaussian<2> target;
    target.mean << yfk.mean[0], yfk.mean[2];
    target.covariance << yfk.covariance(0, 0), yfk.covariance(0, 2),
                         yfk.covariance(2, 0), yfk.covariance(2, 2);
    Gaussian<1> observation;
    observation.mean[0] = yfk.mean[1];
    observation.covariance(0, 0) = yfk.covariance(1, 1);
    if (!(observation.covariance(0, 0) > 0.0)) {
        throw NumericError(NumericStatus::UnsupportedSingularFlight,
                           "deterministic endpoint cannot use midpoint continuous hazard");
    }
    Eigen::Matrix<double, 2, 1> cross;
    cross << yfk.covariance(0, 1), yfk.covariance(2, 1);
    const Gaussian<2> yk = conditionGaussian(target, observation, cross,
                                             Eigen::Matrix<double, 1, 1>::Zero());

    const double varianceK = validateNonnegative(yk.covariance(1, 1), yk.covariance.norm());
    PositiveResult weighted;
    if (varianceK > 0.0) {
        const double coefficient = yk.covariance(0, 1) / varianceK;
        const double residualVariance = validateNonnegative(
            yk.covariance(0, 0) - yk.covariance(0, 1) * coefficient,
            yk.covariance.norm());
        const double intercept = yk.mean[0] - coefficient * yk.mean[1];
        if (residualVariance > 0.0) {
            const double residualStddev = std::sqrt(residualVariance);
            weighted = negativeFluxTimesCdf(yk.mean[1], std::sqrt(varianceK),
                                            intercept / residualStddev,
                                            coefficient / residualStddev);
        } else {
            weighted = negativeFluxTimesAffineIndicator(yk.mean[1], std::sqrt(varianceK),
                                                        intercept, coefficient);
        }
    } else {
        const double flux = std::max(-yk.mean[1], 0.0);
        PositiveResult screen;
        if (yk.covariance(0, 0) > 0.0) {
            screen = positiveFromLog(normalLogCdf(yk.mean[0] /
                                                  std::sqrt(yk.covariance(0, 0))));
        } else {
            screen = yk.mean[0] > 0.0 ? PositiveResult{1.0, 0.0, 0.0, NumericStatus::Degenerate}
                                      : exactZero();
        }
        weighted = flux > 0.0 && screen.status != NumericStatus::ExactZero
                       ? positiveFromLog(std::log(flux) + screen.logValue)
                       : exactZero();
    }
    if (weighted.status == NumericStatus::ExactZero) {
        return {exactZero(), denominator.logValue, -std::numeric_limits<double>::infinity()};
    }
    const double stddevF = std::sqrt(observation.covariance(0, 0));
    const double logPdfF0 = normalLogPdf(-observation.mean[0] / stddevF) - std::log(stddevF);
    const double logJ = logPdfF0 + weighted.logValue;
    return {positiveFromLog(logJ - denominator.logValue), denominator.logValue, logJ};
}

HitStatistics MidpointFlightKernel::hitStatistics(double age) const {
    if (!enableScreen_) {
        Conditional29FlightKernel equivalent(field_, state_);
        return equivalent.hitStatistics(age);
    }
    const Gaussian<5> yfg = ray_.midpointValueGradient(age);
    Gaussian<4> target;
    target.mean << yfg.mean[0], yfg.mean[2], yfg.mean[3], yfg.mean[4];
    target.covariance.setZero();
    const std::array<int, 4> indices{{0, 2, 3, 4}};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        target.covariance(i, j) = yfg.covariance(indices[static_cast<std::size_t>(i)],
                                                indices[static_cast<std::size_t>(j)]);
    }
    Gaussian<1> observation;
    observation.mean[0] = yfg.mean[1];
    observation.covariance(0, 0) = yfg.covariance(1, 1);
    Eigen::Matrix<double, 4, 1> cross;
    for (int i = 0; i < 4; ++i) cross[i] = yfg.covariance(indices[static_cast<std::size_t>(i)], 1);
    const Gaussian<4> conditioned = conditionGaussian(target, observation, cross,
                                                      Eigen::Matrix<double, 1, 1>::Zero());
    Gaussian<3> gradient;
    gradient.mean = conditioned.mean.template segment<3>(1);
    gradient.covariance = conditioned.covariance.template block<3, 3>(1, 1);
    return {gradient, conditioned};
}

} // namespace mf

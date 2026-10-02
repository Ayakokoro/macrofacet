#include "macrofacet/experiments/FirstPassageRice.h"

#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/Quadrature.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mf {
namespace {

double positivePartMean(double mean, double stddev) {
    if (!(stddev > 0.0)) return std::max(0.0, mean);
    const double eta = mean / stddev;
    return stddev * normalPdf(eta) + mean * normalCdf(eta);
}

double gaussianPdf(const DynamicGaussian& gaussian, const Eigen::VectorXd& value,
                   const NumericPolicy& policy) {
    const int dimension = static_cast<int>(gaussian.mean.size());
    if (dimension == 0 || value.size() != dimension ||
        gaussian.covariance.rows() != dimension ||
        gaussian.covariance.cols() != dimension) {
        throw NumericError(NumericStatus::InvalidInput,
                           "Gaussian PDF dimension mismatch");
    }
    const Eigen::MatrixXd covariance =
        0.5 * (gaussian.covariance + gaussian.covariance.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
        if (eig.info() != Eigen::Success ||
            eig.eigenvalues().minCoeff() <
                -covarianceTolerance(std::max(1.0, covariance.norm()), policy)) {
            throw NumericError(NumericStatus::InvalidCovariance,
                               "Rice boundary covariance is not positive semidefinite");
        }
        // A density with respect to full-dimensional Lebesgue measure is not
        // defined for an exactly singular covariance. Rice integration never
        // needs coincident times, so report zero at that measure-zero boundary.
        if (eig.eigenvalues().minCoeff() <=
            covarianceTolerance(std::max(1.0, covariance.norm()), policy)) {
            return 0.0;
        }
        throw NumericError(NumericStatus::InvalidCovariance,
                           "Rice boundary covariance factorization failed");
    }
    const Eigen::VectorXd residual = value - gaussian.mean;
    const Eigen::VectorXd solved = llt.matrixL().solve(residual);
    double logDeterminant = 0.0;
    for (int i = 0; i < dimension; ++i) {
        const double diagonal = llt.matrixL()(i, i);
        if (!(diagonal > 0.0)) return 0.0;
        logDeterminant += 2.0 * std::log(diagonal);
    }
    const double logDensity = -0.5 * (static_cast<double>(dimension) *
        std::log(2.0 * kPi) + logDeterminant + solved.squaredNorm());
    if (logDensity < std::log(std::numeric_limits<double>::min())) return 0.0;
    return std::exp(logDensity);
}

double bivariateNormalPdf(double a, double b, double correlation) {
    const double oneMinusSquared = 1.0 - correlation * correlation;
    if (!(oneMinusSquared > 0.0)) return 0.0;
    const double exponent = -(a * a - 2.0 * correlation * a * b + b * b) /
                            (2.0 * oneMinusSquared);
    if (exponent < std::log(std::numeric_limits<double>::min())) return 0.0;
    return std::exp(exponent) / (2.0 * kPi * std::sqrt(oneMinusSquared));
}

// Plackett's identity: d Phi_2(a,b;rho) / d rho = phi_2(a,b;rho).
double bivariateNormalCdf(double a, double b, double correlation,
                          const NumericPolicy& policy) {
    correlation = std::clamp(correlation, -1.0, 1.0);
    if (correlation >= 1.0 - 8.0 * std::numeric_limits<double>::epsilon()) {
        return normalCdf(std::min(a, b));
    }
    if (correlation <= -1.0 + 8.0 * std::numeric_limits<double>::epsilon()) {
        return std::max(0.0, normalCdf(a) - normalCdf(-b));
    }
    const double independent = normalCdf(a) * normalCdf(b);
    if (correlation == 0.0) return independent;
    const auto integrand = [&](double rho) {
        return bivariateNormalPdf(a, b, rho);
    };
    IntegralResult correction;
    if (correlation > 0.0) {
        correction = integrateFinite(integrand, 0.0, correlation, policy);
    } else {
        correction = integrateFinite(integrand, correlation, 0.0, policy);
        correction.value = -correction.value;
    }
    return std::clamp(independent + correction.value, 0.0,
                      std::min(normalCdf(a), normalCdf(b)));
}

double bivariatePositiveProductMoment(const DynamicGaussian& gaussian,
                                      const NumericPolicy& policy) {
    if (gaussian.mean.size() != 2 || gaussian.covariance.rows() != 2 ||
        gaussian.covariance.cols() != 2) {
        throw NumericError(NumericStatus::InvalidInput,
                           "bivariate positive moment requires dimension two");
    }
    const double variance0 = std::max(0.0, gaussian.covariance(0, 0));
    const double variance1 = std::max(0.0, gaussian.covariance(1, 1));
    const double stddev0 = std::sqrt(variance0);
    const double stddev1 = std::sqrt(variance1);
    if (!(stddev0 > 0.0)) {
        return std::max(0.0, gaussian.mean[0]) *
               positivePartMean(gaussian.mean[1], stddev1);
    }
    if (!(stddev1 > 0.0)) {
        return std::max(0.0, gaussian.mean[1]) *
               positivePartMean(gaussian.mean[0], stddev0);
    }
    double correlation = gaussian.covariance(0, 1) / (stddev0 * stddev1);
    correlation = std::clamp(correlation, -1.0, 1.0);
    const double eta0 = gaussian.mean[0] / stddev0;
    const double eta1 = gaussian.mean[1] / stddev1;
    const double oneMinusSquared = std::max(0.0, 1.0 - correlation * correlation);

    // Very close crossings make the conditional velocity covariance nearly
    // rank one. Directly integrate one standard normal in that limit instead
    // of subtracting nearly equal terms in the closed bivariate expression.
    if (oneMinusSquared < 1e-10) {
        const double lower = -eta0;
        const double conditionalStddev = stddev1 * std::sqrt(oneMinusSquared);
        const auto integrand = [&](double standardNormal) {
            const double z0 = gaussian.mean[0] + stddev0 * standardNormal;
            const double conditionalMean = gaussian.mean[1] +
                stddev1 * correlation * standardNormal;
            return z0 * normalPdf(standardNormal) *
                   positivePartMean(conditionalMean, conditionalStddev);
        };
        return std::max(0.0, integrateSemiInfinite(integrand, lower, policy).value);
    }

    const double root = std::sqrt(oneMinusSquared);
    const double probability = bivariateNormalCdf(eta0, eta1, correlation, policy);
    const double result = stddev0 * stddev1 * (
        (eta0 * eta1 + correlation) * probability +
        eta1 * normalPdf(eta0) * normalCdf((eta1 - correlation * eta0) / root) +
        eta0 * normalPdf(eta1) * normalCdf((eta0 - correlation * eta1) / root) +
        oneMinusSquared * bivariateNormalPdf(eta0, eta1, correlation));
    return std::max(0.0, result);
}

struct StartConditionedDistribution {
    DynamicGaussian values;
    DynamicGaussian velocities;
    Eigen::MatrixXd velocityValueCovariance;
};

StartConditionedDistribution conditionOnStart(
    const FirstPassageStationaryKernel& kernel,
    const std::vector<double>& times,
    double processMean,
    double initialValue,
    const NumericPolicy& policy) {
    const int count = static_cast<int>(times.size());
    DynamicGaussian joint = conditionValueDerivativeGaussian(
        kernel, times, processMean, {0.0}, {initialValue}, policy);
    DynamicGaussian values;
    values.mean = joint.mean.head(count);
    values.covariance = joint.covariance.topLeftCorner(count, count);
    DynamicGaussian velocities;
    velocities.mean = joint.mean.tail(count);
    velocities.covariance = joint.covariance.bottomRightCorner(count, count);
    Eigen::MatrixXd velocityValue = joint.covariance.bottomLeftCorner(count, count);
    return {std::move(values), std::move(velocities), std::move(velocityValue)};
}

DynamicGaussian conditionDownwardSpeedsAtBoundary(
    const StartConditionedDistribution& distribution,
    double threshold,
    const NumericPolicy& policy) {
    const int count = static_cast<int>(distribution.values.mean.size());
    const Eigen::VectorXd boundary = Eigen::VectorXd::Constant(count, threshold);
    DynamicGaussian conditionedVelocity = conditionGaussianDynamic(
        distribution.velocities, distribution.values,
        distribution.velocityValueCovariance, boundary, policy);
    conditionedVelocity.mean = -conditionedVelocity.mean;
    return conditionedVelocity;
}

} // namespace

FirstPassageStationaryKernel::FirstPassageStationaryKernel(
    FirstPassageKernelConfig config)
    : config_(std::move(config)) {}

double FirstPassageStationaryKernel::covariance(double lag) const {
    const double r = std::abs(lag);
    const double x = r / config_.lengthScale;
    if (config_.type == "squared_exponential") {
        return config_.variance * std::exp(-0.5 * x * x);
    }
    if (config_.type == "matern_3_2") {
        const double y = std::sqrt(3.0) * x;
        return config_.variance * (1.0 + y) * std::exp(-y);
    }
    if (config_.type == "matern_5_2") {
        const double y = std::sqrt(5.0) * x;
        return config_.variance * (1.0 + y + y * y / 3.0) * std::exp(-y);
    }
    const double base = 1.0 + x * x / (2.0 * config_.alpha);
    return config_.variance * std::pow(base, -config_.alpha);
}

double FirstPassageStationaryKernel::firstDerivative(double lag) const {
    if (lag < 0.0) return -firstDerivative(-lag);
    const double ell = config_.lengthScale;
    if (config_.type == "squared_exponential") {
        return -lag * covariance(lag) / (ell * ell);
    }
    if (config_.type == "matern_3_2") {
        const double a = std::sqrt(3.0) / ell;
        return -config_.variance * a * a * lag * std::exp(-a * lag);
    }
    if (config_.type == "matern_5_2") {
        const double a = std::sqrt(5.0) / ell;
        return -config_.variance * (a * a / 3.0) * lag * (1.0 + a * lag) *
               std::exp(-a * lag);
    }
    const double base = 1.0 + lag * lag /
        (2.0 * config_.alpha * ell * ell);
    return -config_.variance * lag / (ell * ell) *
           std::pow(base, -config_.alpha - 1.0);
}

double FirstPassageStationaryKernel::secondDerivative(double lag) const {
    const double r = std::abs(lag);
    const double ell = config_.lengthScale;
    if (config_.type == "squared_exponential") {
        return (r * r / (ell * ell * ell * ell) - 1.0 / (ell * ell)) *
               covariance(r);
    }
    if (config_.type == "matern_3_2") {
        const double a = std::sqrt(3.0) / ell;
        return config_.variance * a * a * (a * r - 1.0) * std::exp(-a * r);
    }
    if (config_.type == "matern_5_2") {
        const double a = std::sqrt(5.0) / ell;
        return -config_.variance * (a * a / 3.0) *
               (1.0 + a * r - a * a * r * r) * std::exp(-a * r);
    }
    const double ell2 = ell * ell;
    const double base = 1.0 + r * r / (2.0 * config_.alpha * ell2);
    return -config_.variance / ell2 * std::pow(base, -config_.alpha - 1.0) +
           config_.variance * (config_.alpha + 1.0) * r * r /
               (config_.alpha * ell2 * ell2) *
               std::pow(base, -config_.alpha - 2.0);
}

double FirstPassageStationaryKernel::derivativeVariance() const {
    return -secondDerivative(0.0);
}

DynamicGaussian assembleValueDerivativeGaussian(
    const FirstPassageStationaryKernel& kernel,
    const std::vector<double>& times,
    double processMean) {
    const int count = static_cast<int>(times.size());
    DynamicGaussian result;
    result.mean = Eigen::VectorXd::Zero(2 * count);
    result.mean.head(count).setConstant(processMean);
    result.covariance = Eigen::MatrixXd::Zero(2 * count, 2 * count);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            const double lag = times[static_cast<std::size_t>(i)] -
                               times[static_cast<std::size_t>(j)];
            result.covariance(i, j) = kernel.covariance(lag);
            result.covariance(count + i, j) = kernel.firstDerivative(lag);
            result.covariance(i, count + j) = -kernel.firstDerivative(lag);
            result.covariance(count + i, count + j) =
                -kernel.secondDerivative(lag);
        }
    }
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

DynamicGaussian conditionValueDerivativeGaussian(
    const FirstPassageStationaryKernel& kernel,
    const std::vector<double>& queryTimes,
    double processMean,
    const std::vector<double>& observationTimes,
    const std::vector<double>& observedValues,
    const NumericPolicy& policy) {
    if (observationTimes.size() != observedValues.size()) {
        throw NumericError(NumericStatus::InvalidInput,
                           "process observation time/value dimension mismatch");
    }
    DynamicGaussian target = assembleValueDerivativeGaussian(
        kernel, queryTimes, processMean);
    const int queryCount = static_cast<int>(queryTimes.size());
    const int observationCount = static_cast<int>(observationTimes.size());
    if (observationCount == 0) return target;

    DynamicGaussian observation;
    observation.mean = Eigen::VectorXd::Constant(observationCount, processMean);
    observation.covariance = Eigen::MatrixXd(observationCount, observationCount);
    for (int i = 0; i < observationCount; ++i) {
        for (int j = 0; j < observationCount; ++j) {
            observation.covariance(i, j) = kernel.covariance(
                observationTimes[static_cast<std::size_t>(i)] -
                observationTimes[static_cast<std::size_t>(j)]);
        }
    }
    Eigen::MatrixXd cross(2 * queryCount, observationCount);
    for (int i = 0; i < queryCount; ++i) {
        for (int j = 0; j < observationCount; ++j) {
            const double lag = queryTimes[static_cast<std::size_t>(i)] -
                               observationTimes[static_cast<std::size_t>(j)];
            cross(i, j) = kernel.covariance(lag);
            cross(queryCount + i, j) = kernel.firstDerivative(lag);
        }
    }
    const Eigen::VectorXd observed = Eigen::Map<const Eigen::VectorXd>(
        observedValues.data(), observationCount);
    return conditionGaussianDynamic(target, observation, cross, observed, policy);
}

double riceDowncrossingW1(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double time,
    const NumericPolicy& policy) {
    if (!(time > 0.0)) return 0.0;
    StartConditionedDistribution distribution = conditionOnStart(
        kernel, {time}, processMean, initialValue, policy);
    const Eigen::VectorXd boundary = Eigen::VectorXd::Constant(1, threshold);
    const double density = gaussianPdf(distribution.values, boundary, policy);
    if (!(density > 0.0)) return 0.0;
    const DynamicGaussian downwardSpeeds = conditionDownwardSpeedsAtBoundary(
        distribution, threshold, policy);
    const double stddev = std::sqrt(std::max(
        0.0, downwardSpeeds.covariance(0, 0)));
    return density * positivePartMean(downwardSpeeds.mean[0], stddev);
}

double riceDowncrossingW2(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double firstTime,
    double secondTime,
    const NumericPolicy& policy) {
    if (!(firstTime > 0.0) || !(secondTime > firstTime)) return 0.0;
    // The two boundary constraints become the same constraint on the diagonal.
    // Below the double-precision covariance resolution, evaluate the finite
    // one-sided limit at the nearest resolvable separation. This is especially
    // important for Matérn 3/2, whose velocity covariance has a cusp at zero.
    const double relativeSeparation = kernel.config().type == "matern_3_2"
        ? 1e-4
        : 64.0 * std::sqrt(std::numeric_limits<double>::epsilon());
    const double minimumSeparation = relativeSeparation *
        kernel.config().lengthScale;
    if (secondTime - firstTime < minimumSeparation &&
        secondTime > minimumSeparation) {
        firstTime = secondTime - minimumSeparation;
    }
    StartConditionedDistribution distribution = conditionOnStart(
        kernel, {firstTime, secondTime}, processMean, initialValue, policy);
    const Eigen::VectorXd boundary = Eigen::VectorXd::Constant(2, threshold);
    const double density = gaussianPdf(distribution.values, boundary, policy);
    if (!(density > 0.0)) return 0.0;
    const DynamicGaussian downwardSpeeds = conditionDownwardSpeedsAtBoundary(
        distribution, threshold, policy);
    return density * bivariatePositiveProductMoment(
        downwardSpeeds, policy);
}

IntegralResult riceSecondOrderCorrection(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double time,
    const NumericPolicy& policy) {
    if (!(time > 0.0)) return {};
    const auto integrand = [&](double firstTime) {
        try {
            return riceDowncrossingW2(kernel, processMean, threshold, initialValue,
                                      firstTime, time, policy);
        } catch (const NumericError& error) {
            throw NumericError(error.status(), "W2(" + std::to_string(firstTime) +
                ", " + std::to_string(time) + ") failed: " + error.what());
        }
    };
    return integrateFinite(integrand, 0.0, time, policy);
}

} // namespace mf

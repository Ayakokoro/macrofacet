#pragma once

#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/mathutility/NumericPolicy.h"
#include "macrofacet/mathutility/SmallGaussian.h"
#include <vector>

namespace mf {

// Scalar stationary covariance interface used by the first-passage experiment.
// The derivatives are with respect to the signed lag r in k(r).
class FirstPassageStationaryKernel {
public:
    explicit FirstPassageStationaryKernel(FirstPassageKernelConfig config);

    double covariance(double lag) const;
    double firstDerivative(double lag) const;
    double secondDerivative(double lag) const;
    double derivativeVariance() const;

    const FirstPassageKernelConfig& config() const { return config_; }

private:
    FirstPassageKernelConfig config_;
};

// Assemble the Gaussian vector
// [X(t_1), ..., X(t_n), X'(t_1), ..., X'(t_n)].
DynamicGaussian assembleValueDerivativeGaussian(
    const FirstPassageStationaryKernel& kernel,
    const std::vector<double>& times,
    double processMean);

// Condition the assembled value/derivative vector on arbitrary exact value
// observations X(s_j)=x_j. The implementation uses the common PSD-aware
// Schur-complement conditioner from SmallGaussian.h.
DynamicGaussian conditionValueDerivativeGaussian(
    const FirstPassageStationaryKernel& kernel,
    const std::vector<double>& queryTimes,
    double processMean,
    const std::vector<double>& observationTimes,
    const std::vector<double>& observedValues,
    const NumericPolicy& policy = defaultNumericPolicy());

// Rice product intensities for downward crossings, conditional on X(0)=initialValue.
// W2 is the unintegrated two-time product intensity and requires 0<t1<t2.
double riceDowncrossingW1(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double time,
    const NumericPolicy& policy = defaultNumericPolicy());

double riceDowncrossingW2(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double firstTime,
    double secondTime,
    const NumericPolicy& policy = defaultNumericPolicy());

// A_2(t) = integral_0^t W2(u,t) du, i.e. the second correction in
// f_tau(t) = W1(t) - A_2(t) + ... .
IntegralResult riceSecondOrderCorrection(
    const FirstPassageStationaryKernel& kernel,
    double processMean,
    double threshold,
    double initialValue,
    double time,
    const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

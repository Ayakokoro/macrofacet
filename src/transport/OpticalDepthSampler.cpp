#include "macrofacet/transport/OpticalDepthSampler.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/mathutility/RootFinding.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {
namespace {

std::vector<double> integrationKnots(const FlightKernel& kernel, double a, double b,
                                      int initialCells, const NumericPolicy& policy) {
    std::vector<double> knots{a};
    if (a == b) return knots;
    const Vector3& w = kernel.state().direction;
    const auto& field = kernel.field();
    const double precision = w.dot(field.kernel.precision() * w);
    double step = (b - a) / initialCells;
    if (precision > 0.0) step = std::min(step, 0.25 / std::sqrt(precision));
    const double voxelSize = field.mean->voxelSizeHint();
    if (voxelSize > 0.0) {
        step = std::min(step, voxelSize);
    } else {
        const double meanScale = field.mean->bounds(field.activeDomain).maximumGradientNorm;
        if (meanScale > 0.0) step = std::min(step, 0.5 * field.kernel.sigma() / meanScale);
    }
    const double countReal = std::ceil((b - a) / step);
    if (!std::isfinite(countReal) || countReal > policy.maxQuadratureSubdivisions)
        throw NumericError(NumericStatus::IntegrationNotConverged, "initial optical-depth partition exceeds budget");
    const int count = std::max(1, static_cast<int>(countReal));
    for (int i = 1; i <= count; ++i) knots.push_back(i == count ? b : a + (b - a) * i / count);
    field.mean->appendRayBreakpoints(kernel.state().birthPosition, w, a, b, knots);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    return knots;
}

IntegralResult integrateInterval(const FlightKernel& kernel, double a, double b,
                                const NumericPolicy& policy, TrackingDiagnostics* diagnostics) {
    auto integrand = [&](double t) {
        if (diagnostics) ++diagnostics->hazardEvaluations;
        const double value = kernel.evaluate(t).hazard.value;
        if (!(value >= 0.0) || !std::isfinite(value))
            throw NumericError(NumericStatus::InvalidInput, "hazard must be finite and nonnegative");
        return value;
    };
    const auto result = integrateFinite(integrand, a, b, policy);
    if (diagnostics) diagnostics->quadratureIntervals += result.subdivisions;
    if (result.status != NumericStatus::Ok)
        throw NumericError(result.status, "optical-depth quadrature did not converge");
    return result;
}

} // namespace

PositiveResult integrateHazard(const FlightKernel& kernel, double a, double b,
                               const NumericPolicy& policy) {
    if (!(a >= kernel.currentAge() && b >= a && b <= kernel.maximumAgeInDomain()))
        throw std::out_of_range("hazard integration interval lies outside flight domain");
    if (a == b) return exactZero();
    const auto knots = integrationKnots(kernel, a, b, 16, policy);
    double sum = 0.0, error = 0.0;
    for (std::size_t i = 1; i < knots.size(); ++i) {
        NumericPolicy local = policy;
        local.absoluteTolerance *= (knots[i] - knots[i - 1]) / (b - a);
        const auto part = integrateInterval(kernel, knots[i - 1], knots[i], local, nullptr);
        sum += part.value;
        error += part.absError;
    }
    if (sum == 0.0) return exactZero();
    return {sum, std::log(sum), error, NumericStatus::Ok};
}

OpticalDepthSampler::OpticalDepthSampler(const FlightKernel& kernel, const NumericPolicy& policy,
                                       double maximumAge, int initialCells,
                                       TrackingDiagnostics* diagnostics)
    : kernel_(kernel), policy_(policy), begin_(kernel.currentAge()),
      end_(maximumAge < 0.0 ? kernel.maximumAgeInDomain() : maximumAge), diagnostics_(diagnostics) {
    if (!(end_ >= begin_ && end_ <= kernel.maximumAgeInDomain()) || initialCells < 1)
        throw std::invalid_argument("invalid regular tracking interval or partition");
    knots_ = integrationKnots(kernel_, begin_, end_, initialCells, policy_);
}

IntegralResult OpticalDepthSampler::integrate(double a, double b) const {
    NumericPolicy local = policy_;
    // Allocate a small part of the residual budget to quadrature, apportioned
    // by distance. Relative errors add in proportion to positive segment mass.
    local.relativeTolerance *= 0.001;
    local.absoluteTolerance *= 0.001 * (b - a) / (end_ - begin_);
    return integrateInterval(kernel_, a, b, local, diagnostics_);
}

FlightSample OpticalDepthSampler::sample(Random& rng) {
    return invert(-std::log1p(-rng.openUniform01()));
}

FlightSample OpticalDepthSampler::invert(double target) {
    if (!(target > 0.0) || !std::isfinite(target))
        throw std::invalid_argument("optical target must be finite and positive");
    std::size_t index = 0;
    double prefix = 0.0, prefixError = 0.0;
    while (index + 1 < knots_.size()) {
        if (index == segments_.size()) {
            const auto part = integrate(knots_[index], knots_[index + 1]);
            segments_.push_back({knots_[index], knots_[index + 1], prefix, prefixError,
                                 part.value, part.absError});
        }
        const auto& segment = segments_[index];
        const double next = segment.prefix + segment.depth;
        const double error = segment.prefixError + segment.error;
        if (target < next) {
            const double l0 = segment.lo;
            const double base = segment.prefix;
            double lastError = 0.0;
            auto evaluate = [&](double t) -> RootEvaluation {
                IntegralResult part;
                if (t == segment.hi) part = {segment.depth, segment.error, 0, NumericStatus::Ok};
                else if (t != l0) part = integrate(l0, t);
                if (diagnostics_) ++diagnostics_->hazardEvaluations;
                const double rate = kernel_.evaluate(t).hazard.value;
                lastError = segment.prefixError + part.absError;
                return {base + part.value, rate, lastError};
            };
            const double initial = l0 + (target - base) / segment.depth * (segment.hi - l0);
            int bisections = 0;
            const auto root = solveMonotoneSafeguardedNewton(evaluate, target, l0, segment.hi,
                                                            initial, policy_, &bisections);
            if (diagnostics_) {
                diagnostics_->newtonIterations += root.iterations;
                diagnostics_->bisectionSteps += bisections;
                diagnostics_->maximumResidual = std::max(diagnostics_->maximumResidual, std::abs(root.residual));
                diagnostics_->maximumIntegrationError = std::max(diagnostics_->maximumIntegrationError, lastError);
            }
            if (root.status != NumericStatus::Ok)
                throw NumericError(root.status, "safeguarded Newton optical-depth inverse did not converge");
            const auto rate = kernel_.evaluate(root.value).hazard;
            if (!(rate.value > 0.0))
                throw NumericError(NumericStatus::NeedHigherPrecision, "inverse reached a zero-hazard interval");
            const double actualDepth = target + root.residual;
            return {true, root.value, -actualDepth, rate.logValue - actualDepth, std::nullopt};
        }
        prefix = next;
        prefixError = error;
        ++index;
    }
    if (diagnostics_) diagnostics_->maximumIntegrationError =
        std::max(diagnostics_->maximumIntegrationError, prefixError);
    if (target - prefix < prefixError)
        throw NumericError(NumericStatus::NeedHigherPrecision, "ambiguous optical-depth escape boundary");
    return {false, end_, -prefix, std::nullopt, std::exp(-prefix)};
}

FlightSample sampleFlight(const FlightKernel& kernel, Random& rng, const NumericPolicy& policy,
                          TrackingDiagnostics* diagnostics, int initialCells,
                          double maximumAge) {
    const double target = -std::log1p(-rng.openUniform01());
    NumericPolicy working = policy;
    for (int attempt = 0; ; ++attempt) {
        try {
            OpticalDepthSampler sampler(kernel, working, maximumAge, initialCells, diagnostics);
            return sampler.invert(target);
        } catch (const NumericError& error) {
            if (attempt == 2 || (error.status() != NumericStatus::NeedHigherPrecision &&
                                 error.status() != NumericStatus::IntegrationNotConverged)) throw;
            working.absoluteTolerance *= 0.1;
            working.relativeTolerance *= 0.1;
        }
    }
}

} // namespace mf

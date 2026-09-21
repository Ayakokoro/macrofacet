#include "macrofacet/transport/OpticalDepthSampler.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/mathutility/RootFinding.h"
#include <cmath>
#include <limits>

namespace mf {

PositiveResult integrateHazard(const FlightKernel& kernel, double a, double b,
                               const NumericPolicy& policy) {
    if (!(a >= kernel.currentAge() && b >= a && b <= kernel.maximumAgeInDomain())) {
        throw std::out_of_range("hazard integration interval lies outside flight domain");
    }
    if (a == b) return exactZero();
    auto integrand = [&](double t) {
        const double value = kernel.evaluate(t).hazard.value;
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw NumericError(NumericStatus::InvalidInput, "hazard must be finite and nonnegative");
        }
        return value;
    };
    const IntegralResult integral = integrateFinite(integrand, a, b, policy);
    if (integral.status == NumericStatus::IntegrationNotConverged) {
        throw NumericError(integral.status, "hazard integration did not converge");
    }
    if (!(integral.value > 0.0)) return exactZero();
    return {integral.value, std::log(integral.value), integral.absError, integral.status};
}

FlightSample sampleFlight(const FlightKernel& kernel, Random& rng,
                          const NumericPolicy& policy) {
    const double a = kernel.currentAge();
    const double b = kernel.maximumAgeInDomain();
    const double opticalTarget = -std::log1p(-rng.openUniform01());
    const PositiveResult total = integrateHazard(kernel, a, b, policy);
    const double totalOpticalDepth = total.status == NumericStatus::ExactZero ? 0.0 : total.value;
    if (opticalTarget >= totalOpticalDepth) {
        return {false, b, -totalOpticalDepth, std::nullopt, std::exp(-totalOpticalDepth)};
    }
    auto cumulative = [&](double t) {
        const PositiveResult value = integrateHazard(kernel, a, t, policy);
        return value.status == NumericStatus::ExactZero ? 0.0 : value.value;
    };
    const RootResult root = solveMonotoneIncreasing(cumulative, opticalTarget, a, b, policy);
    if (root.status != NumericStatus::Ok) {
        throw NumericError(root.status, "failed to invert cumulative optical depth");
    }
    const PositiveResult hazard = kernel.evaluate(root.value).hazard;
    if (hazard.status == NumericStatus::ExactZero) {
        throw NumericError(NumericStatus::NeedHigherPrecision,
                           "optical-depth inverse landed in a zero-hazard interval");
    }
    return {true, root.value, -opticalTarget, hazard.logValue - opticalTarget, std::nullopt};
}

} // namespace mf


#pragma once

#include "macrofacet/mathutility/NumericPolicy.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace mf {

inline RootResult solveMonotoneIncreasing(const std::function<double(double)>& f,
                                          double target, double lo, double hi,
                                          const NumericPolicy& policy = defaultNumericPolicy()) {
    double flo = f(lo) - target;
    double fhi = f(hi) - target;
    if (!std::isfinite(flo) || !std::isfinite(fhi) || flo > 0.0 || fhi < 0.0) {
        return {lo, flo, 0, NumericStatus::RootNotBracketed};
    }
    if (flo == 0.0) return {lo, 0.0, 0, NumericStatus::Ok};
    if (fhi == 0.0) return {hi, 0.0, 0, NumericStatus::Ok};
    double mid = lo;
    double fm = flo;
    for (int iteration = 1; iteration <= policy.maxRootIterations; ++iteration) {
        mid = lo + 0.5 * (hi - lo);
        fm = f(mid) - target;
        if (!std::isfinite(fm)) {
            throw NumericError(NumericStatus::InvalidInput, "non-finite root function");
        }
        const double xTolerance = policy.absoluteTolerance +
                                  policy.relativeTolerance * std::max(1.0, std::abs(mid));
        if (fm == 0.0 || hi - lo <= 2.0 * xTolerance) {
            return {mid, fm, iteration, NumericStatus::Ok};
        }
        if (fm < 0.0) lo = mid;
        else hi = mid;
    }
    return {mid, fm, policy.maxRootIterations, NumericStatus::NeedHigherPrecision};
}

inline RootResult expandAndSolveNegativeDomain(const std::function<double(double)>& f,
                                               double target, double upper,
                                               const NumericPolicy& policy = defaultNumericPolicy()) {
    double width = 1.0;
    double lower = upper - width;
    for (int i = 0; i < 1024 && f(lower) > target; ++i) {
        width *= 2.0;
        lower = upper - width;
        if (!std::isfinite(lower)) break;
    }
    return solveMonotoneIncreasing(f, target, lower, upper, policy);
}

} // namespace mf


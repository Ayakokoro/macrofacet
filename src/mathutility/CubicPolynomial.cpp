#include "macrofacet/mathutility/CubicPolynomial.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {
HermitePolynomial hermitePolynomial(double value0, double derivative0,
                                    double value1, double derivative1, double width) {
    return {2.0 * value0 - 2.0 * value1 + width * (derivative0 + derivative1),
            -3.0 * value0 + 3.0 * value1 - width * (2.0 * derivative0 + derivative1),
            width * derivative0, value0};
}

std::vector<double> hermiteKnots(const HermitePolynomial& polynomial) {
    std::vector<double> result{0.0, 1.0};
    // Normalize before finding extrema: multiplying a polynomial by a tiny
    // positive number must not hide its interior crossings.
    const double scale = std::max({std::abs(polynomial.a), std::abs(polynomial.b),
                                   std::abs(polynomial.c)});
    if (scale == 0.0) return result;
    const double qa = 3.0 * (polynomial.a / scale);
    const double qb = 2.0 * (polynomial.b / scale);
    const double qc = polynomial.c / scale;
    if (std::abs(qa) <= 1e-14) {
        if (std::abs(qb) > 1e-14) {
            const double root = -qc / qb;
            if (root > 0.0 && root < 1.0) result.push_back(root);
        }
    } else {
        const double discriminant = qb * qb - 4.0 * qa * qc;
        if (discriminant >= 0.0) {
            const double root = std::sqrt(std::max(0.0, discriminant));
            const double q = -0.5 * (qb + std::copysign(root, qb));
            const double r0 = q / qa;
            const double r1 = q == 0.0 ? r0 : qc / q;
            if (r0 > 0.0 && r0 < 1.0) result.push_back(r0);
            if (r1 > 0.0 && r1 < 1.0) result.push_back(r1);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end(), [](double a, double b) {
        return std::abs(a - b) <= 1e-14;
    }), result.end());
    return result;
}

double hermiteMinimum(const HermitePolynomial& polynomial) {
    double result = std::numeric_limits<double>::infinity();
    for (double s : hermiteKnots(polynomial)) result = std::min(result, polynomial.value(s));
    return result;
}

HermiteCrossing firstHermiteDowncrossing(const HermitePolynomial& polynomial,
                                         double width, double tolerance) {
    const std::vector<double> knots = hermiteKnots(polynomial);
    for (std::size_t i = 1; i < knots.size(); ++i) {
        double left = knots[i - 1];
        double right = knots[i];
        const double leftValue = polynomial.value(left);
        const double rightValue = polynomial.value(right);
        // At q=0 the process starts on the boundary with positive derivative;
        // that prescribed birth point is not itself a new first passage.
        if (!(leftValue > 0.0 || (left == 0.0 && polynomial.derivative(0.0) > 0.0)) ||
            rightValue > 0.0) {
            continue;
        }
        // An interior stationary point touching zero is not a downcrossing.
        if (right < 1.0 && rightValue == 0.0) continue;
        double lo = left;
        double hi = right;
        for (int iteration = 0; iteration < 80 && (hi - lo) * width > tolerance; ++iteration) {
            const double middle = 0.5 * (lo + hi);
            if (polynomial.value(middle) > 0.0) lo = middle;
            else hi = middle;
        }
        const double root = 0.5 * (lo + hi);
        const double derivative = polynomial.derivative(root) / width;
        if (derivative < 0.0) return {true, root, derivative};
    }
    return {};
}


} // namespace mf

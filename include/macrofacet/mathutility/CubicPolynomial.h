#pragma once

#include <vector>

namespace mf {

// Polynomial in the local coordinate u in [0,1]. Derivatives here are d/du.
struct HermitePolynomial {
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    double value(double u) const { return ((a*u+b)*u+c)*u+d; }
    double derivative(double u) const { return (3.0*a*u+2.0*b)*u+c; }
};

HermitePolynomial hermitePolynomial(double value0, double derivative0,
    double value1, double derivative1, double width);
std::vector<double> hermiteKnots(const HermitePolynomial& polynomial);
double hermiteMinimum(const HermitePolynomial& polynomial);

struct HermiteCrossing {
    bool found = false;
    double fraction = 0.0;
    double derivative = 0.0; // d/dx, not d/du
};

// Searches every monotone subinterval, including hidden interior crossings.
// A prescribed zero at u=0 with outward derivative is not a new hit.
HermiteCrossing firstHermiteDowncrossing(const HermitePolynomial& polynomial,
    double width, double tolerance);

} // namespace mf

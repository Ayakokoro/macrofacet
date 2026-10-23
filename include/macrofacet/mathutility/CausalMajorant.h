#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define MF_MAJORANT_HD __host__ __device__
#else
#define MF_MAJORANT_HD
#endif

namespace mf::causal_majorant {
// Lower Bernstein control bounds for p(t)=a*t^3+b*t^2+c*t+d, t in [0,1].
// The derivative is with respect to q, with t=(q-lo)/width. Outward
// arithmetic and a forward-error allowance also cover Horner evaluation.
MF_MAJORANT_HD inline double addDown(double x,double y) {
#if defined(__CUDA_ARCH__)
    return __dadd_rd(x,y);
#else
    return std::nextafter(x+y,-INFINITY);
#endif
}
MF_MAJORANT_HD inline double divideDown(double x,double y) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rd(x,y);
#else
    return std::nextafter(x/y,-INFINITY);
#endif
}
struct LowerBounds { double value, derivative; };
MF_MAJORANT_HD inline LowerBounds cubicLowerBounds(double a,double b,double c,double d,double width) {
    const double p0=d,p1=addDown(d,divideDown(c,3));
    const double p2=addDown(d,divideDown(addDown(addDown(c,c),b),3));
    const double p3=addDown(addDown(addDown(d,c),b),a);
    const double v0=c,v1=addDown(c,b);
    const double v2=addDown(addDown(c,addDown(b,b)),addDown(addDown(a,a),a));
    constexpr double allowance=64*2.22044604925031308085e-16;
    const double valueError=allowance*(1+fabs(a)+fabs(b)+fabs(c)+fabs(d));
    const double derivativeError=allowance*(1+(3*fabs(a)+2*fabs(b)+fabs(c))/width);
    return {addDown(fmin(fmin(p0,p1),fmin(p2,p3)),-valueError),
        addDown(divideDown(fmin(v0,fmin(v1,v2)),width),-derivativeError)};
}
// Analytic upper bound on phi(b)/Phi(b) * [phi(v)-v*Phi(-v)].
// For x>=0, the Gaussian Mills ratio R(x)>=2/(x+sqrt(x*x+4)).
// See https://dlmf.nist.gov/7.8.E2, after changing to standard-normal units.
// This avoids cancellation in the positive-v tail and erfcx in bound setup.
// b>=0: Phi(b)>=1/2. v<=0: Phi(-v)<=1.
MF_MAJORANT_HD inline double localRateUpper(double b,double v) {
    constexpr double invRoot2Pi=.39894228040143267794;
    constexpr double tiny=2.22507385850720138309e-308;
    const double mills=b<=0 ? .5*(-b+hypot(b,2.)) : 2*invRoot2Pi*exp(-.5*b*b);
    const double phi=invRoot2Pi*exp(-.5*v*v);
    const double t=v>0 ? 2/(v+hypot(v,2.)) : 0;
    const double negativeMean=v>0 ? phi*t*t : phi-v;
    // Elementary-function rounding and underflow allowance, independent of
    // the analytical slack above. Candidate rates are checked against M.
    return (mills*negativeMean)*(1+1e-12)+tiny;
}
} // namespace mf::causal_majorant
#undef MF_MAJORANT_HD

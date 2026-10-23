#include "TestHarness.h"
#include "macrofacet/mathutility/CausalMajorant.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <array>
#include <random>

namespace {
double referenceRate(double b,double v) {
    // Independent Gaussian functions, not the CUDA cancellation-free formula.
    return mf::normalPdfOverCdf(b)*std::max(0.,mf::normalPdf(v)-v*mf::normalCdf(-v));
}
}
void testCausalMajorant(TestContext& context) {
    using namespace mf::causal_majorant;
    for (double b:{-100.,-26.,-8.,-1.,-1e-12,0.,1e-12,1.,8.,26.,100.})
        for (double v:{-100.,-26.,-8.,-1.,-1e-12,0.,1e-12,1.,8.,26.,100.}) {
            const double bound=localRateUpper(b,v);
            context.require(std::isfinite(bound) && bound>0 && bound>=referenceRate(b,v),
                "analytic local hazard majorant covers Gaussian tails and sign changes");
        }
    std::mt19937_64 rng(20261022);
    std::uniform_real_distribution<double> value(-8,8);
    for (int i=0;i<2000;++i) {
        // Includes linear/constant cells, nearly coincident voxel boundaries,
        // and cubics with internal extrema; no endpoint-only maximum assumption.
        const double a=i%3 ? value(rng) : 0,b=i%5 ? value(rng) : 0;
        const double c=i%7 ? value(rng) : 0,d=value(rng);
        const double width=std::pow(10.,-(i%15));
        const auto lo=cubicLowerBounds(a,b,c,d,width);
        const double bound=localRateUpper(lo.value,lo.derivative);
        for (int j=0;j<=16;++j) {
            const double t=j/16.,p=((a*t+b)*t+c)*t+d,v=((3*a*t+2*b)*t+c)/width;
            context.require(p>=lo.value && v>=lo.derivative,"Bernstein lower bounds cover cubic value and derivative");
            context.require(referenceRate(p,v)<=bound,"cell majorant covers local hazard including interior extrema");
        }
    }
    const auto constant=cubicLowerBounds(0,0,0,0,1);
    context.require(localRateUpper(constant.value,constant.derivative)>=1/mf::kPi,
        "constant zero mean cell bounds analytic exponential rate 1/pi");
}

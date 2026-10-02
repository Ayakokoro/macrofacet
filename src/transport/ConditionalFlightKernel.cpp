#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/mathutility/SmallGaussian.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

namespace mf {
namespace {

constexpr double boundGuard = 1.0 + 1e-11;

double upward(double x) {
    if (!(x >= 0.0) || !std::isfinite(x))
        throw NumericError(NumericStatus::InvalidMajorant, "non-finite conditional majorant");
    const double result=std::nextafter(x * boundGuard, std::numeric_limits<double>::infinity());
    if (!std::isfinite(result))
        throw NumericError(NumericStatus::InvalidMajorant, "conditional majorant overflow");
    return result;
}

// Positive series: truncation after 20 terms is below double roundoff for u<=1.
// The bounds below allow for both this remainder and floating-point arithmetic.
std::pair<double,double> scaledExponentialRemainders(double u) {
    double e2=0.5, term=0.5;
    for (int n=1; n<=20; ++n) { term *= u/(n+2); e2 += term; }
    double e4=1.0/12.0;
    term=e4;
    for (int n=1; n<=12; ++n) {
        term *= u*u/((2*n+3.0)*(2*n+4.0)); e4 += term;
    }
    return {e2,e4};
}

struct SeFactors { double valueVariance, slopeVariance, regression; };

SeFactors seFactors(double a, double t) {
    const double u=a*t*t;
    if (u<=1.0) {
        const auto e=scaledExponentialRemainders(u);
        return {std::exp(-u)*u*u*e.first, u*u*e.second/e.first,
                1.0/(t*e.first)};
    }
    const double r=std::exp(-u);
    if (r==0.0) return {1.0,1.0,0.0};
    const double v=-std::expm1(-u)-u*r;
    return {v, (1.0-(2.0+u*u)*r+r*r)/v, a*t*u*r/v};
}

double slopeStddevBound(double sigma2, double a, double end) {
    const double u=a*end*end;
    if (u<=1.0) {
        const auto e=scaledExponentialRemainders(u);
        // E4(u) increases, E2(u)>=1/2.
        return upward(std::sqrt(sigma2*a)*u*std::sqrt(2.0*e.second));
    }
    return upward(std::sqrt(sigma2*a));
}

std::pair<double,double> quadraticRange(double c0, double c1, double c2,
                                        double lo, double hi) {
    const auto value=[&](double t) { return c0+t*(c1+t*c2); };
    double lower=std::min(value(lo),value(hi)), upper=std::max(value(lo),value(hi));
    if (c2!=0.0) {
        const double root=-c1/(2.0*c2);
        if (root>lo && root<hi) { lower=std::min(lower,value(root)); upper=std::max(upper,value(root)); }
    }
    const double error=64.0*std::numeric_limits<double>::epsilon()*
        (std::abs(c0)+std::abs(c1)*hi+std::abs(c2)*hi*hi);
    return {lower-error,upper+error};
}

std::pair<double,double> productRange(double a, double b, double c, double d) {
    return {std::min({a*c,a*d,b*c,b*d}),std::max({a*c,a*d,b*c,b*d})};
}

double envelopeMaximum(double logC1, double c2, double power, double exponent, double end) {
    // Maximize C1 t^-power exp(-c2/t^exponent), including its zero limit at birth.
    if (!(c2>0.0)) throw NumericError(NumericStatus::InvalidMajorant, "birth envelope lost positive separation");
    const double t=std::min(end,std::pow(exponent*c2/power,1.0/exponent));
    const double logValue=logC1-power*std::log(t)-c2/std::pow(t,exponent);
    return upward(std::exp(logValue));
}

double momentIntervalMajorant(double fmin, double fmax, double kmin,
                              double sigma2, double a, double lo, double hi) {
    const auto left=seFactors(a,lo), right=seFactors(a,hi);
    const double sdmin=std::sqrt(sigma2*left.valueVariance)/boundGuard;
    const double sdmax=upward(std::sqrt(sigma2*right.valueVariance));
    const double zmin=fmin/(fmin>=0.0 ? sdmax : sdmin);
    const double muMin=kmin-std::max(left.regression*fmax,right.regression*fmax);
    const double smax=slopeStddevBound(sigma2,a,hi);
    // For positive conditional slope retain the Gaussian suppression of the
    // negative flux. The earlier s/sqrt(2*pi) bound lost this suppression.
    const double logFlux=muMin>=0.0
        ? std::log(smax)+normalLogPdf(muMin/smax)+std::log(boundGuard)
        : std::log(upward(-muMin+smax*kInvSqrtTwoPi));
    const double logDensity=zmin>=0.0
        ? std::log(2.0)+normalLogPdf(zmin)-std::log(sdmin)
        : std::log1p(-zmin)-std::log(sdmin);
    return upward(std::exp(logDensity+logFlux));
}

} // namespace

FlightState startConditionalExterior(const GPSSField& field, const Point3& entry,
                                     const Vector3& direction, Random& rng) {
    const PointPrior prior = field.pointPrior(entry);
    const double sigma = std::sqrt(prior.varianceF);
    // At one point the SE value and gradient are independent.
    const double logTail = normalLogCdf(prior.meanF / sigma);
    const double z = -normalQuantileFromLogCdf(logTail +
                                                std::log(rng.openUniform01()));
    FlightState state = startExternalFlight(entry, direction);
    state.birthValue = std::max(std::nextafter(0.0, 1.0), prior.meanF + sigma * z);
    state.birthGradient = sampleGaussianPSD(
        Gaussian<3>{prior.meanG, prior.covarianceG}, rng);
    return state;
}

ConditionalFlightKernel::ConditionalFlightKernel(const GPSSField& field,
                                                 const FlightState& state)
    : FlightKernel(field, state) {
    if (!field.kernel.supportsAnalyticConditionalTransport()) {
        throw std::invalid_argument(
            std::string("global_conditional is not implemented for kernel '") +
            covarianceKernelTypeName(field.kernel.type()) +
            "'; the common KernelJet interface is available for a future conditioner");
    }
    if (!state.birthGradient.allFinite() || !(state.birthValue>=0.0) || !std::isfinite(state.birthValue))
        throw std::invalid_argument("conditional flight requires a finite birth observation");
    birthMean_=field_.mean->evaluate(state.birthPosition);
    deltaGradient_=state.birthGradient-birthMean_.gradient;
    deltaValue_=state.birthValue-birthMean_.value;
    deltaSlope_=state.direction.dot(deltaGradient_);
    precisionDirection_=field.kernel.precision()*state.direction;
    inverseLength2_=state.direction.dot(precisionDirection_);
    sigma2_=field.kernel.sigma()*field.kernel.sigma();
    if (!(inverseLength2_>0.0))
        throw NumericError(NumericStatus::UnsupportedSingularFlight, "conditional flight requires positive ray precision");
}

ConditionalScalarStatistics ConditionalFlightKernel::scalarStatistics(double age) const {
    if (!(age > 0.0 && age <= maximumAgeInDomain()))
        throw std::out_of_range("conditional flight age outside domain");
    const double u=inverseLength2_*age*age, r=std::exp(-0.5*u);
    const MeanJet target=field_.mean->evaluate(state_.birthPosition+age*state_.direction);
    const double meanF=u<1e-3
        ? state_.birthValue+age*deltaSlope_+
          field_.mean->valueDifference(state_.birthPosition,age*state_.direction)+
          std::expm1(-0.5*u)*(deltaValue_+age*deltaSlope_)
        : target.value+r*(deltaValue_+age*deltaSlope_);
    const double meanK=state_.direction.dot(target.gradient)+
        r*((1.0-u)*deltaSlope_-inverseLength2_*age*deltaValue_);
    const SeFactors factors=seFactors(inverseLength2_,age);
    return {meanF,sigma2_*factors.valueVariance,meanK,
            meanK-factors.regression*meanF,sigma2_*inverseLength2_*factors.slopeVariance};
}

HazardEvaluation ConditionalFlightKernel::evaluate(double age) const {
    if (age == 0.0) return {exactZero()};
    const auto m = scalarStatistics(age);
    if (!(m.varianceF > 0.0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"conditional variance underflow");
    const PositiveResult flux = negativePartMean(m.meanAtZero, std::sqrt(m.varianceAtZero));
    if (flux.status == NumericStatus::ExactZero) return {exactZero()};
    const double sd = std::sqrt(m.varianceF);
    const double z = m.meanF / sd;
    const double logRatio=z < -8.0 ? std::log(normalPdfOverCdf(z))
                                  : normalLogPdf(z)-normalLogCdf(z);
    return {positiveFromLog(logRatio - std::log(sd) +
                            flux.logValue)};
}

HitStatistics ConditionalFlightKernel::hitStatistics(double age) const {
    const auto m=scalarStatistics(age);
    const auto factors=seFactors(inverseLength2_,age);
    const double u=inverseLength2_*age*age, r=std::exp(-0.5*u);
    const auto target=field_.mean->evaluate(state_.birthPosition+age*state_.direction);
    Gaussian<3> g;
    g.mean=target.gradient+r*deltaGradient_-precisionDirection_*
        (age*r*(deltaValue_+age*deltaSlope_)+factors.regression*m.meanF/inverseLength2_);
    const Matrix3 longitudinal=precisionDirection_*precisionDirection_.transpose()/inverseLength2_;
    g.covariance=sigma2_*(-std::expm1(-u)*(field_.kernel.precision()-longitudinal)+
                         factors.slopeVariance*longitudinal);
    g.covariance = 0.5 * (g.covariance + g.covariance.transpose());
    return {g};
}

double ConditionalFlightKernel::intervalMajorant(double lo, double hi) const {
    if (!(lo>0.0 && hi>lo && hi<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional bound interval");
    const double a=inverseLength2_;
    const MeanRayBounds mb=field_.mean->rayBounds(state_.birthPosition,state_.direction,lo,hi);
    if (!mb.certified) throw NumericError(NumericStatus::InvalidMajorant,"uncertified ray mean bounds");
    const double rmin=std::exp(-0.5*a*hi*hi), rmax=std::exp(-0.5*a*lo*lo);
    const auto offset=quadraticRange(deltaValue_,deltaSlope_,0.0,lo,hi);
    const auto correction=productRange(rmin,rmax,offset.first,offset.second);
    double fmin=mb.minimumValue+correction.first;
    double fmax=mb.maximumValue+correction.second;
    const auto slope=quadraticRange(deltaSlope_,-a*deltaValue_,-a*deltaSlope_,lo,hi);
    const auto slopeCorrection=productRange(rmin,rmax,slope.first,slope.second);
    const double kmin=mb.minimumDerivative+slopeCorrection.first;
    const double kmax=mb.maximumDerivative+slopeCorrection.second;
    // Bound the mean from its midpoint and a certified derivative range to
    // preserve cancellation between the prior and the birth correction.
    const double middle=0.5*lo+0.5*hi;
    const double um=a*middle*middle;
    const double targetValue=field_.mean->evaluate(state_.birthPosition+middle*state_.direction).value;
    const double center=um<1e-3
        ? state_.birthValue+middle*deltaSlope_+
          field_.mean->valueDifference(state_.birthPosition,middle*state_.direction)+
          std::expm1(-0.5*um)*(deltaValue_+middle*deltaSlope_)
        : targetValue+std::exp(-0.5*um)*(deltaValue_+middle*deltaSlope_);
    const double roundoff=256.0*std::numeric_limits<double>::epsilon()*
        (std::abs(targetValue)+std::abs(birthMean_.value)+std::abs(deltaValue_)+
         std::abs(middle*deltaSlope_)+1e-30);
    const double radius=0.5*(hi-lo)*std::max(std::abs(kmin),std::abs(kmax))*boundGuard+roundoff;
    fmin=std::max(fmin,center-radius);
    fmax=std::min(fmax,center+radius);
    return momentIntervalMajorant(fmin,fmax,kmin,sigma2_,a,lo,hi);
}

ConditionalBirthMajorant ConditionalFlightKernel::birthMajorant(double end) const {
    if (!(end>=currentAge() && end<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional majorant limit");
    ConditionalBirthMajorant result;
    if (end==0.0) return result;
    const double a=inverseLength2_, ell=1.0/std::sqrt(a), f0=state_.birthValue;
    // The first Taylor interval must lie inside one smooth interpolation cell.
    double smoothEnd=std::min(end,0.5*ell);
    std::vector<double> knots;
    field_.mean->appendRayBreakpoints(state_.birthPosition,state_.direction,0.0,smoothEnd,knots);
    if (!knots.empty()) smoothEnd=0.5*(*std::min_element(knots.begin(),knots.end()));
    const MeanRayBounds birth=field_.mean->rayBounds(state_.birthPosition,state_.direction,0.0,smoothEnd);
    if (!birth.certified || !std::isfinite(birth.maximumSecondDerivative))
        throw NumericError(NumericStatus::InvalidMajorant,"mean lacks a certified smooth birth interval; bake to NanoVDB");
    const double k=birth.beginDerivative+deltaSlope_;
    if (f0==0.0 && !(k>0.0))
        throw NumericError(NumericStatus::UnsupportedSingularFlight,"surface birth must enter the positive side");
    const double curvature=birth.maximumSecondDerivative;
    result.smoothEnd=smoothEnd;
    result.slope=k;
    result.curvature=curvature;
    const auto remainder=[&](double h) {
        return upward(0.5*(curvature+a*(std::abs(deltaValue_)+h*std::abs(deltaSlope_))));
    };
    const auto slopeRemainder=[&](double h) {
        return upward(curvature+a*std::abs(deltaValue_)+1.5*a*h*std::abs(deltaSlope_));
    };
    double h=f0==0.0 ? std::min(smoothEnd,k/(std::sqrt(sigma2_)*a)) : smoothEnd;
    for (int count=0;;++count) {
        const double error=remainder(h);
        if (f0==0.0 ? error*h<=0.5*k : std::abs(k)*h+error*h*h<=0.5*f0) break;
        h*=0.5;
        if (count>=1024 || !(h>0.0))
            throw NumericError(NumericStatus::InvalidMajorant,"cannot certify conditional birth segment");
    }
    result.split=h;
    const double u=a*h*h, error=remainder(h);
    const auto e=scaledExponentialRemainders(u);
    const double vmin=0.5*sigma2_*a*a*std::exp(-u)/boundGuard;
    const double vmax=upward(sigma2_*a*a*e.first);
    const double kabs=std::abs(k)+slopeRemainder(h)*h;
    const double smax=slopeStddevBound(sigma2_,a,h);
    const double common=kabs+2.0*(std::abs(k)+error*h)+smax*kInvSqrtTwoPi;
    if (f0==0.0) {
        const double b=(k-error*h)/boundGuard;
        result.maximum=envelopeMaximum(std::log(std::sqrt(2.0/kPi)*common/std::sqrt(vmin)),
                                           b*b/(2.0*vmax),2.0,2.0,h);
    } else {
        const double b=(f0-std::abs(k)*h-error*h*h)/boundGuard;
        const double fluxCoefficient=2.0*f0+h*common;
        result.maximum=envelopeMaximum(std::log(std::sqrt(2.0/kPi)*fluxCoefficient/std::sqrt(vmin)),
                                           b*b/(2.0*vmax),3.0,4.0,h);
    }
    return result;
}

double ConditionalFlightKernel::birthAdjacentMajorant(
    double lo, double hi, const ConditionalBirthMajorant& birth) const {
    if (!(lo>0.0 && hi>lo && hi<=birth.smoothEnd))
        throw NumericError(NumericStatus::InvalidInput, "invalid birth-adjacent interval");
    const double a=inverseLength2_;
    const double rem=upward(0.5*(birth.curvature+a*(std::abs(deltaValue_)+
                                                     hi*std::abs(deltaSlope_))));
    const double slopeRem=upward(birth.curvature+a*std::abs(deltaValue_)+
                                 1.5*a*hi*std::abs(deltaSlope_));
    return momentIntervalMajorant(
        quadraticRange(state_.birthValue,birth.slope,-rem,lo,hi).first,
        quadraticRange(state_.birthValue,birth.slope, rem,lo,hi).second,
        birth.slope-slopeRem*hi,sigma2_,a,lo,hi);
}

ConditionalMajorants ConditionalFlightKernel::twoSegmentMajorants(double end) const {
    const ConditionalBirthMajorant birth=birthMajorant(end);
    ConditionalMajorants result;
    result.split=birth.split;
    result.nearMaximum=birth.maximum;
    if (end==0.0) return result;
    const double a=inverseLength2_, ell=1.0/std::sqrt(a), h=birth.split;

    // Bound the entire far segment. Internal intervals only tighten the proof:
    // their maximum is the default far Poisson rate. Retain the intervals for
    // exact thinning when this constant would cause excessive null events.
    const auto intervalMaximum=[&](double lo, double hi) {
        ++result.boundIntervals;
        if (hi>birth.smoothEnd) return intervalMajorant(lo,hi);
        return birthAdjacentMajorant(lo,hi,birth);
    };
    for (double lo=h; lo<end;) {
        double hi=std::min(end,lo+std::min(0.5*lo,0.5*ell));
        if (!(hi>lo)) throw NumericError(NumericStatus::InvalidMajorant,"conditional bound interval cannot advance");
        double bound=intervalMaximum(lo,hi);
        // Refine expensive near-birth enclosures, where vF grows like t^4.
        // The work threshold only chooses between already valid bounds; it
        // does not limit the rate, discard an interval, or probe the hazard.
        while (lo<ell && hi-lo>lo/16.0 && bound*(end-h)>256.0) {
            const double refinedHi=lo+std::max(lo/16.0,0.5*(hi-lo));
            if (!(refinedHi<hi)) break;
            hi=refinedHi;
            bound=intervalMaximum(lo,hi);
        }
        result.farMaximum=std::max(result.farMaximum,bound);
        result.farIntervals.push_back({lo,hi,bound});
        lo=hi;
    }
    return result;
}

} // namespace mf

#include "macrofacet/gpss/Matern32Reference.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
using State2 = Eigen::Vector2d;
using Matrix2 = Eigen::Matrix2d;

Matern32Transition matern32Transition(double step) {
    if (!(step>0.0) || !std::isfinite(step)) throw std::invalid_argument("invalid Matern transition step");
    constexpr double lambda = 1.0; // rho(x)=(1+x)exp(-x)
    const double r = lambda * step;
    const double exponential = std::exp(-r);
    Matern32Transition result;
    result.matrix << exponential * (1.0 + r), exponential * step,
                     -exponential * lambda * r, exponential * (1.0 - r);

    const double x = 2.0 * r;
    double q00 = 0.0;
    if (x < 1.0) {
        // 1-exp(-x)(1+x+x^2/2), evaluated without the O(x^3)
        // cancellation that would otherwise dominate at refinement scales.
        double term = x * x * x / 6.0;
        double tail = term;
        for (int order = 4; order < 64; ++order) {
            term *= x / static_cast<double>(order);
            tail += term;
            if (term <= 2e-16 * std::max(1.0, tail)) break;
        }
        q00 = std::exp(-x) * tail;
    } else {
        q00 = 1.0 - std::exp(-x) * (1.0 + x + 0.5 * x * x);
    }
    const double q01 = 2.0 * lambda * r * r * std::exp(-2.0 * r);
    const double q11 = lambda * lambda *
        (-std::expm1(-2.0 * r) + std::exp(-2.0 * r) * (2.0 * r - 2.0 * r * r));
    result.covariance << q00, q01, q01, q11;
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

namespace {
State2 sampleGaussian2(const State2& mean, const Matrix2& source, Random& rng) {
    Matrix2 covariance = 0.5 * (source + source.transpose());
    const double scale = std::max({std::abs(covariance(0, 0)),
                                  std::abs(covariance(0, 1)),
                                  std::abs(covariance(1, 1)),
                                  std::numeric_limits<double>::min()});
    const double tolerance = 256.0 * std::numeric_limits<double>::epsilon() * scale;
    if (covariance(0, 0) < -tolerance || covariance(1, 1) < -tolerance) {
        throw std::runtime_error("negative Matérn state covariance diagonal");
    }
    covariance(0, 0) = std::max(0.0, covariance(0, 0));
    covariance(1, 1) = std::max(0.0, covariance(1, 1));
    State2 sample = mean;
    if (covariance(0, 0) > 0.0) {
        const double l00 = std::sqrt(covariance(0, 0));
        const double l10 = covariance(1, 0) / l00;
        double remainder = covariance(1, 1) - l10 * l10;
        if (remainder < -tolerance) {
            throw std::runtime_error("Matérn state covariance is not PSD");
        }
        remainder = std::max(0.0, remainder);
        const double z0 = rng.standardNormal();
        const double z1 = rng.standardNormal();
        sample[0] += l00 * z0;
        sample[1] += l10 * z0 + std::sqrt(remainder) * z1;
    } else {
        if (std::abs(covariance(0, 1)) > tolerance) {
            throw std::runtime_error("singular Matérn covariance has nonzero cross term");
        }
        sample[1] += std::sqrt(covariance(1, 1)) * rng.standardNormal();
    }
    return sample;
}

} // namespace

State2 advanceMatern32(const State2& state, double step, Random& rng) {
    const Matern32Transition transition = matern32Transition(step);
    return sampleGaussian2(transition.matrix * state, transition.covariance, rng);
}

Matern32Bridge matern32BridgeMidpoint(const State2& left, const State2& right,
                                            double width) {
    const double half = 0.5 * width;
    const Matern32Transition first = matern32Transition(half);
    const Matern32Transition second = matern32Transition(width - half);
    const Matern32Transition total = matern32Transition(width);
    const Matrix2 cross = first.covariance * second.matrix.transpose();
    Eigen::LDLT<Matrix2> factor(total.covariance);
    if (factor.info() != Eigen::Success) {
        throw std::runtime_error("Matérn bridge covariance factorization failed");
    }
    const Matrix2 gain = factor.solve(cross.transpose()).transpose();
    Matern32Bridge result;
    result.mean = first.matrix * left +
        gain * (right - total.matrix * left);
    result.covariance = first.covariance - gain * cross.transpose();
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

Matern32State initializeMatern32(const RayMeanProfile& mean,
    const RayStartCondition& start, Random& rng) {
    const auto& first=mean.segments().front();
    const double b=first.value(0.0);
    if (start.mode==RayStartMode::SurfaceOutward) {
        if (!(start.outwardDerivative>0.0) || !std::isfinite(start.outwardDerivative))
            throw std::invalid_argument("surface birth requires a positive finite total derivative");
        return {-b,start.outwardDerivative-first.derivative(0.0)};
    }
    if (start.mode!=RayStartMode::PositiveExterior)
        throw std::invalid_argument("unknown ray start mode");
    // P(Z>-b)=Phi(b). Work in the log tail, also for deep negative means.
    const double z=-normalQuantileFromLogCdf(normalLogCdf(b)+std::log(rng.openUniform01()));
    return {std::max(z,std::nextafter(-b,std::numeric_limits<double>::infinity())),
            rng.standardNormal()};
}

namespace {
struct Crossing { bool hit=false; double x=0.0, speed=0.0; };

Crossing inspect(const RayMeanSegment& mean, const FirstPassageReferenceSettings& settings,
    double x0, const State2& state0, double x1, const State2& state1,
    int depth, bool surfaceBirth, double birthSlope, Random& rng,
    FirstPassageReferenceSample& diagnostics) {
    diagnostics.deepestRefinement=std::max(diagnostics.deepestRefinement,depth);
    const double width=x1-x0;
    auto polynomial=hermitePolynomial(mean.value(x0)+state0[0],
        mean.derivative(x0)+state0[1],mean.value(x1)+state1[0],
        mean.derivative(x1)+state1[1],width);
    if (surfaceBirth && x0==0.0) {
        polynomial.d=0.0;
        polynomial.c=width*birthSlope;
    }
    const auto coarse=firstHermiteDowncrossing(polynomial,width,settings.crossingTolerance);
    if (width<=settings.minimumStep || depth>=settings.maximumRefinementDepth) {
        return coarse.found ? Crossing{true,x0+coarse.fraction*width,-coarse.derivative}
                            : Crossing{};
    }
    const auto bridge=matern32BridgeMidpoint(state0,state1,width);
    const double middle=0.5*x0+0.5*x1;
    if (!(middle>x0 && middle<x1)) throw std::runtime_error("reference interval cannot refine");
    const double sd=std::sqrt(std::max(0.0,bridge.covariance(0,0)));
    // This is a numerical refinement criterion, not a certified probability
    // bound on missed continuous crossings. Validate reference resolution.
    if (!coarse.found && hermiteMinimum(polynomial)>settings.bridgeSigmaMargin*sd &&
        mean.value(middle)+bridge.mean[0]>settings.bridgeSigmaMargin*sd) return {};
    ++diagnostics.bridgeRefinements;
    const State2 midpoint=sampleGaussian2(bridge.mean,bridge.covariance,rng);
    const auto left=inspect(mean,settings,x0,state0,middle,midpoint,depth+1,
                           surfaceBirth,birthSlope,rng,diagnostics);
    if (left.hit) return left;
    return inspect(mean,settings,middle,midpoint,x1,state1,depth+1,
                   surfaceBirth,birthSlope,rng,diagnostics);
}
}

FirstPassageReferenceSample sampleMatern32FirstPassage(const RayMeanProfile& mean,
    const RayStartCondition& start, const FirstPassageReferenceSettings& settings, Random& rng) {
    if (!(settings.step>0.0) || !std::isfinite(settings.step) ||
        !(settings.minimumStep>0.0 && settings.minimumStep<=settings.step) ||
        !(settings.crossingTolerance>0.0 && settings.crossingTolerance<settings.minimumStep) ||
        !(settings.bridgeSigmaMargin>0.0) || !std::isfinite(settings.bridgeSigmaMargin) ||
        settings.maximumRefinementDepth<0 ||
        std::ldexp(settings.step,-settings.maximumRefinementDepth)>settings.minimumStep)
        throw std::invalid_argument("invalid first-passage reference resolution");
    FirstPassageReferenceSample result;
    result.distance=mean.maximumX();
    State2 state=initializeMatern32(mean,start,rng);
    const auto partition=mean.subdivided(settings.step);
    for (const auto& segment : partition.segments()) {
        const double width=segment.end-segment.begin;
        const State2 next=advanceMatern32(state,width,rng);
        ++result.transitions;
        const auto crossing=inspect(segment,settings,segment.begin,state,segment.end,next,0,
            start.mode==RayStartMode::SurfaceOutward,start.outwardDerivative,rng,result);
        if (crossing.hit) {
            result.hit=true;
            result.distance=crossing.x;
            result.speed=crossing.speed;
            return result;
        }
        state=next;
    }
    return result;
}

} // namespace mf


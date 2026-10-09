#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/gpss/RayMeanProfile.h"
#include <cstdint>

namespace mf {

// Renewal+ convention: rho(x)=(1+x)exp(-x), Var(Z)=Var(Z')=1.
using Matern32State = Eigen::Vector2d;
struct Matern32Transition {
    Eigen::Matrix2d matrix;
    Eigen::Matrix2d covariance;
};
struct Matern32Bridge {
    Matern32State mean;
    Eigen::Matrix2d covariance;
};
Matern32Transition matern32Transition(double step);
Matern32State advanceMatern32(const Matern32State& state, double step, Random& rng);
Matern32Bridge matern32BridgeMidpoint(const Matern32State& left,
    const Matern32State& right, double width);

enum class RayStartMode { PositiveExterior, SurfaceOutward };
struct RayStartCondition {
    RayStartMode mode = RayStartMode::PositiveExterior;
    // Known total derivative ell/sigma * dot(omega,g0), mode B only.
    double outwardDerivative = 0.0;
};

struct FirstPassageReferenceSettings {
    double step = 1.0/64.0;
    double minimumStep = 1.0/1024.0;
    double crossingTolerance = 1e-9;
    double bridgeSigmaMargin = 6.0;
    int maximumRefinementDepth = 12;
};
struct FirstPassageReferenceSample {
    bool hit = false;
    double distance = 0.0; // normalized x; horizon for a censored sample
    double speed = std::numeric_limits<double>::quiet_NaN(); // W>0 on hit
    std::uint64_t transitions = 0, bridgeRefinements = 0;
    int deepestRefinement = 0;
};

// Mode A samples hidden initial values; they must never become neural inputs.
Matern32State initializeMatern32(const RayMeanProfile& mean,
    const RayStartCondition& start, Random& rng);

// Numerical continuous-first-passage reference: exact node/bridge transitions,
// cubic Hermite crossing localization. Requires step-size convergence checks.
FirstPassageReferenceSample sampleMatern32FirstPassage(const RayMeanProfile& mean,
    const RayStartCondition& start, const FirstPassageReferenceSettings& settings,
    Random& rng);

} // namespace mf

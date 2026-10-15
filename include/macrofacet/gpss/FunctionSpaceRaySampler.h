#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/gpss/RayMeanProfile.h"
#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace mf {

// Unit-variance, unit-length SE residual. All lengths are q = distance / ell.
struct FunctionSpaceSamplerConfig {
    int historyPoints = 1;
    int batchSize = 8;
    double step = 1.0 / 64.0;
    double rankTolerance = 1e-10;
    std::string crossingRule = "tungsten_linear";
};

struct FunctionSpaceState { double value = 0.0, derivative = 0.0; };
struct FunctionSpaceHistory {
    std::vector<double> values;
    double derivative = 0.0; // Only the latest longitudinal derivative survives.
};

enum class FunctionSpaceStart { Unconditioned, PositiveExterior, SurfaceOutward };

struct FunctionSpaceFirstPassageSample {
    double distance = 0.0;
    bool hit = false;
    double speed = 0.0;
    int initialSign = 1;
    std::uint64_t generatedPoints = 0;
};

// Immutable covariance plans can be shared between trajectories and threads.
// Both crossing rules share joint value/analytic-derivative draws;
// only the last H values and the last derivative condition the following batch.
// tungsten_linear uses total-field endpoint secants, Hermite is an optional reference.
class FunctionSpaceRaySampler {
public:
    explicit FunctionSpaceRaySampler(FunctionSpaceSamplerConfig config);
    const FunctionSpaceSamplerConfig& config() const { return config_; }
    std::vector<FunctionSpaceState> next(FunctionSpaceHistory& history, Random& rng) const;

private:
    struct Plan { Eigen::MatrixXd regression, squareRoot; };
    FunctionSpaceSamplerConfig config_;
    std::vector<Plan> plans_;
};

FunctionSpaceFirstPassageSample sampleFunctionSpaceFirstPassage(
    const FunctionSpaceRaySampler& sampler, const RayMeanProfile& mean,
    FunctionSpaceStart start, double outwardDerivative, Random& rng);

} // namespace mf

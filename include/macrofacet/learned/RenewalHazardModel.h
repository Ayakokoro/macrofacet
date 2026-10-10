#pragma once

#include "macrofacet/gpss/Matern32Reference.h"
#include "macrofacet/gpss/CovarianceKernel.h"
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mf {

class RenewalBatchSession;
struct RenewalNetworkWeights;

// Each component is normalized separately on W > 0. No Rice reweighting.
struct RenewalSpeedMixture {
    std::vector<double> weights, means, scales;
    double sample(Random& rng) const;
};

double samplePositiveNormal(double mean, double scale, Random& rng);

// Immutable CPU inference for a trained unit-decay Matern-3/2 or unit-length SE model.
// Weights/activations are float32; geometry and cumulative integration are double.
class RenewalHazardModel {
public:
    using State = Eigen::VectorXf;
    struct Step {
        std::array<double, 4> rates;
        State nextState; // used only after surviving the complete segment
    };
    static RenewalHazardModel load(const std::filesystem::path& bundle);
    State initialize(const RayMeanSegment& first, const RayStartCondition& start) const;
    Step evaluate(const State& enteringState, const RayMeanSegment& segment) const;
    bool hasMixture() const;
    RenewalSpeedMixture mixture(const State& enteringState,
        const RayMeanSegment& segment, double u) const;
    int hiddenSize() const;
    CovarianceKernelType kernelType() const;
    const std::string& checkpointSha256() const;
private:
    friend class RenewalBatchSession;
    std::shared_ptr<const RenewalNetworkWeights> batchWeights() const;
    struct Impl;
    State encodeSegment(const RayMeanSegment& segment) const;
    explicit RenewalHazardModel(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

} // namespace mf

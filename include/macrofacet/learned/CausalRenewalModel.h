#pragma once
#include "macrofacet/learned/RenewalHazardModel.h"

namespace mf {
struct CausalNetworkWeights;

struct CausalRenewalPrediction {
    double q = 0, cumulativeHazard = 0, logHazard = 0;
    RenewalSpeedMixture mixture;
};

struct CausalRenewalSample {
    bool hit = false;
    double q = 0, speed = 0; // normalized distance; W > 0 on a hit
    std::size_t intervals = 0;
};

// A: F(0)>0. B: F(0)=0 with known outward derivative. Both sample the next
// downcrossing. The immutable network is shared; each call owns its GRU state.
class CausalRenewalModel {
public:
    static CausalRenewalModel load(const std::filesystem::path& path);
    static CausalRenewalModel load(std::istream& input);
    double step() const;
    double integrationStep() const;
    const std::string& checkpointSha256() const;
    // Immutable weights shared with optional native inference backends.
    const CausalNetworkWeights& networkWeights() const;
    std::vector<CausalRenewalPrediction> query(const RayMeanProfile& profile,
        const RayStartCondition& start, const std::vector<double>& q) const;
    // Uniforms: exponential optical depth, mixture component, component CDF.
    CausalRenewalSample sample(const RayMeanProfile& profile, const RayStartCondition& start,
        const std::array<double,3>& uniforms, double normalizedTolerance = 1e-9) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    explicit CausalRenewalModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
};
} // namespace mf

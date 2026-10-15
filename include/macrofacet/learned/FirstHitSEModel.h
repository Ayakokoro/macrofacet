#pragma once
#include "macrofacet/learned/RenewalHazardModel.h"
#include <array>
#include <memory>

namespace mf {

struct FirstHitSEPrediction {
    double q=0, logTransmittance=0, logDensity=0;
    std::array<double,2> cumulative{}, logHazard{}, hitBranchProbability{};
    std::array<RenewalSpeedMixture,2> mixture;
};
struct FirstHitSESample {
    bool hit=false;
    double q=0, speed=0; // q from medium entry; W is positive on either branch.
    int initialSign=1;
};

// Immutable scalar inference for RNO, GRU and fixed-profile spline models.
// All integrate out the unobserved entry sign, with branch order {+1,-1}.
class FirstHitSEModel {
public:
    static FirstHitSEModel load(const std::filesystem::path& path);
    std::vector<FirstHitSEPrediction> query(const RayMeanProfile& profile,
        const std::vector<double>& q, double maximumStep=0) const;
    FirstHitSESample sample(const RayMeanProfile& profile, Random& rng,
        double maximumStep=0, double normalizedTolerance=1e-7) const;
    const std::string& architecture() const;
    double maximumStep() const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    explicit FirstHitSEModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
};
} // namespace mf

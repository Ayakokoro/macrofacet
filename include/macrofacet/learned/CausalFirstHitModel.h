#pragma once
#include "macrofacet/learned/FirstHitSEModel.h"

namespace mf {
// Fixed-grid first-zero model. These uniforms have a fixed meaning independent
// of batching, interval count and the eventual hit/escape outcome.
using CausalFirstHitUniforms = std::array<double,4>;
CausalFirstHitUniforms causalFirstHitUniforms(Random& rng);

class CausalFirstHitSession;
class CausalFirstHitModel {
public:
    static CausalFirstHitModel load(const std::filesystem::path& path);
    double step() const;
    std::vector<FirstHitSEPrediction> query(const RayMeanProfile& profile,
        const std::vector<double>& q) const;
    FirstHitSESample sample(const RayMeanProfile& profile, const CausalFirstHitUniforms& uniforms,
        double tolerance=1e-7) const;
private:
    friend class CausalFirstHitSession;
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    explicit CausalFirstHitModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
};

// A session is owned by one render thread. Weights and hidden states stay on the
// selected device; completed/escaped slots never enter subsequent GRU updates.
class CausalFirstHitSession {
public:
    CausalFirstHitSession(const CausalFirstHitModel&, const std::string& backend="scalar", int cpuThreads=1);
    ~CausalFirstHitSession();
    CausalFirstHitSession(const CausalFirstHitSession&)=delete;
    CausalFirstHitSession& operator=(const CausalFirstHitSession&)=delete;
    std::vector<FirstHitSESample> sample(const std::vector<const RayMeanProfile*>& profiles,
        const std::vector<CausalFirstHitUniforms>& uniforms, double tolerance=1e-7);
    static void validateBackend(const std::string& backend);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mf

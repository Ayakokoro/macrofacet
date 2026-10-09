#pragma once
#include "macrofacet/learned/RenewalHazardModel.h"

namespace mf {

// One session owns a bounded set of independent ray slots. It is not shared
// between threads; the immutable scalar model can still be shared as before.
class RenewalBatchSession {
public:
    RenewalBatchSession(const RenewalHazardModel& model, int capacity, const std::string& backend);
    ~RenewalBatchSession();
    RenewalBatchSession(const RenewalBatchSession&) = delete;
    RenewalBatchSession& operator=(const RenewalBatchSession&) = delete;
    void initialize(const std::vector<int>& slots, const std::vector<RayMeanSegment>& first,
                    const std::vector<RayStartCondition>& starts);
    // Advances each listed slot exactly once. Unlisted slots retain their state.
    std::vector<std::array<double,4>> evaluate(const std::vector<int>& slots,
                                              const std::vector<RayMeanSegment>& segments);
    // Uses the ENTERING state and encoding cached by that slot's last evaluate,
    // even though evaluate has already advanced its recurrent state.
    std::vector<RenewalSpeedMixture> mixture(const std::vector<int>& slots,
                                            const std::vector<double>& u);
    const std::string& backend() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool renewalCudaAvailable(std::string* reason = nullptr);
bool renewalTorchAvailable();
std::string resolveRenewalBackend(const std::string& requested);

} // namespace mf

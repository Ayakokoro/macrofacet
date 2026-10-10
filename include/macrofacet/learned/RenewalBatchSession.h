#pragma once
#include "macrofacet/learned/RenewalHazardModel.h"
#include <atomic>
#include <cstdint>
#include <optional>

namespace mf {

struct ExperimentConfig;
struct RenderedImage;

struct RenewalBatchRequests {
    std::vector<int> initializeSlots, segmentSlots, mixtureSlots;
    std::vector<RayMeanSegment> firstSegments, segments;
    std::vector<RayStartCondition> starts;
    std::vector<double> coordinates;
};
struct RenewalBatchResults {
    std::vector<std::array<double,4>> rates;
    std::vector<RenewalSpeedMixture> mixtures;
};

// One session owns a bounded set of independent ray slots. It is not shared
// between threads; the immutable scalar model can still be shared as before.
class RenewalBatchSession {
public:
    using Ticket = std::uint64_t;
    RenewalBatchSession(const RenewalHazardModel& model, int capacity, const std::string& backend,
                        int maximumInFlight = 2);
    ~RenewalBatchSession();
    RenewalBatchSession(const RenewalBatchSession&) = delete;
    RenewalBatchSession& operator=(const RenewalBatchSession&) = delete;
    // One upload and one readback for all ready requests. Initialization may
    // precede the SAME slot's first segment in this submission. Mixture slots
    // must be disjoint from both other groups, preserving their entering context.
    // All requests are validated before modifying device state.
    RenewalBatchResults submit(const RenewalBatchRequests& requests);
    // CUDA returns after enqueuing work, without waiting for the result. CPU
    // completes inference inline. Every touched slot is locked until collect.
    // Requests may be destroyed/modified once submitAsync returns. Tickets are
    // session-local, consumed exactly once; at most maximumInFlight may exist.
    Ticket submitAsync(const RenewalBatchRequests& requests);
    bool isReady(Ticket ticket) const;
    std::optional<RenewalBatchResults> tryCollect(Ticket ticket);
    RenewalBatchResults collect(Ticket ticket); // waits only for this ticket
    std::size_t pendingCount() const;
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
    friend RenderedImage renderRenewalWavefront(const ExperimentConfig&,
        std::atomic<std::uint64_t>*);
    // Only the wavefront scheduler may supply trusted slot IDs and ownership.
    // IDs are in range and unique per group; segment state is initialized and
    // mixture requests have a cached step and are disjoint from both groups.
    // It must collect the previous ticket before submitting another batch.
    // Debug uses full validation; Release skips slot validation and bookkeeping.
    // Workers fill only their owned slots before publishing ready. Storage is
    // fixed for the frame, and a published slot stays immutable until collect.
    struct WavefrontFeatures {
        std::vector<std::array<double,4>> initial;
        std::vector<std::array<double,5>> segment;
        std::vector<std::array<double,2>> mixture; // value, db/dx in normalized ray distance
    };
    static std::array<double,4> initialFeatures(const RayMeanSegment&, const RayStartCondition&);
    static std::array<double,2> mixtureFeatures(const RayMeanSegment&, double coordinate);
    Ticket submitWavefrontAsync(const RenewalBatchRequests& requests, const WavefrontFeatures& features);
    // Retains high-water storage; requests define active result prefixes. Only
    // call while the destination is exclusively owned by the inference thread.
    void collectInto(Ticket ticket, RenewalBatchResults& result);
    template<bool ValidateSlots, bool Precomputed>
    Ticket submitAsyncImpl(const RenewalBatchRequests& requests, const WavefrontFeatures* features);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool renewalCudaAvailable(std::string* reason = nullptr);
bool renewalTorchAvailable();
std::string resolveRenewalBackend(const std::string& requested);

} // namespace mf

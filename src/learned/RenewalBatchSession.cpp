#include "macrofacet/learned/RenewalBatchSession.h"
#include "RenewalBatchBackend.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {
double finiteInput(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("nonfinite Renewal batch input");
    return x;
}
}

bool renewalTorchAvailable() {
#ifdef MACROFACET_HAS_TORCH
    return true;
#else
    return false;
#endif
}
bool renewalCudaAvailable(std::string* reason) {
#ifdef MACROFACET_HAS_TORCH
    return queryRenewalTorchCuda(reason);
#else
    if (reason) *reason = "built without LibTorch (enable MACROFACET_ENABLE_TORCH)";
    return false;
#endif
}
std::string resolveRenewalBackend(const std::string& requested) {
    if (requested == "scalar") return requested;
    if (requested == "auto") {
        if (renewalCudaAvailable()) return "torch_cuda";
        return renewalTorchAvailable() ? "torch_cpu" : "scalar";
    }
    if (requested == "torch_cpu" && renewalTorchAvailable()) return requested;
    if (requested == "torch_cuda") {
        std::string reason;
        if (renewalCudaAvailable(&reason)) return requested;
        throw std::runtime_error("Renewal torch_cuda unavailable: " + reason);
    }
    if (requested == "torch_cpu")
        throw std::runtime_error("Renewal torch_cpu requires MACROFACET_ENABLE_TORCH=ON");
    throw std::invalid_argument("unknown Renewal backend: " + requested);
}

struct RenewalBatchSession::Impl {
    int capacity;
    std::string name;
    std::shared_ptr<const RenewalNetworkWeights> weights;
    std::unique_ptr<RenewalBatchBackend> engine;
    std::vector<unsigned char> status; // 0: uninitialized, 1: ready, 2: cached step
    std::vector<RayMeanSegment> last;
    std::vector<unsigned char> seen, busy;
    struct Pending {
        Ticket ticket = 0; // Zero marks reusable storage, not a live ticket.
        bool checkedSlots = true;
        std::vector<int> initialSlots, segmentSlots, mixtureSlots, touched;
        std::vector<RayMeanSegment> segments;
        std::vector<double> derivatives;
        void clear() {
            ticket = 0;
            initialSlots.clear(); segmentSlots.clear(); mixtureSlots.clear(); touched.clear();
            segments.clear(); derivatives.clear();
        }
    };
    std::vector<Pending> pending;
    Ticket nextTicket = 1;
    std::size_t find(Ticket ticket) const {
        for (std::size_t i = 0; i < pending.size(); ++i)
            if (ticket && pending[i].ticket == ticket) return i;
        throw std::invalid_argument("invalid or already collected Renewal ticket");
    }
    void release(Pending& p, bool failed) {
        if (p.checkedSlots) for (int id : p.touched) busy[id] = 0;
        if (failed) {
            for (int id : p.initialSlots) status[id] = 0;
            for (int id : p.segmentSlots) status[id] = 0;
        }
        p.clear(); // Keep vector capacities across submissions, including failures.
    }

};

RenewalBatchSession::RenewalBatchSession(const RenewalHazardModel& model, int capacity,
                                       const std::string& backend, int maximumInFlight) : impl_(std::make_unique<Impl>()) {
    if (capacity < 1 || capacity > 65536) throw std::invalid_argument("Renewal batch capacity must be 1..65536");
    if (maximumInFlight < 1 || maximumInFlight > 4)
        throw std::invalid_argument("Renewal maximum in-flight batches must be 1..4");
    impl_->pending.resize(maximumInFlight); impl_->busy.resize(capacity,0);
    impl_->capacity = capacity;
    impl_->name = resolveRenewalBackend(backend);
    if (impl_->name == "scalar") throw std::invalid_argument("scalar backend has no batch session");
    impl_->weights = model.batchWeights();
    impl_->status.resize(capacity, 0); impl_->last.resize(capacity); impl_->seen.resize(capacity);
#ifdef MACROFACET_HAS_TORCH
    impl_->engine = makeRenewalTorchBackend(*impl_->weights, capacity, impl_->name == "torch_cuda", maximumInFlight);
#endif
}
RenewalBatchSession::~RenewalBatchSession() = default;
const std::string& RenewalBatchSession::backend() const { return impl_->name; }

std::array<double,4> RenewalBatchSession::initialFeatures(
    const RayMeanSegment& first, const RayStartCondition& start) {
    if (!(first.end > first.begin)) throw std::invalid_argument("invalid Renewal initial segment");
    const double b = first.value(first.begin);
    if (start.mode == RayStartMode::SurfaceOutward) {
        if (!(start.outwardDerivative > 0) || !std::isfinite(start.outwardDerivative))
            throw std::invalid_argument("Renewal surface start requires positive finite derivative");
        return {1,b,-b,start.outwardDerivative-first.derivative(first.begin)};
    }
    if (start.mode != RayStartMode::PositiveExterior)
        throw std::invalid_argument("invalid Renewal start mode");
    return {0,b,0,0};
}

std::array<double,2> RenewalBatchSession::mixtureFeatures(const RayMeanSegment& segment, double u) {
    if (!(u >= 0 && u <= 1)) throw std::invalid_argument("invalid Renewal batch mixture coordinate");
    const double db = segment.polynomial.derivative(u)/(segment.end-segment.begin);
    return {segment.polynomial.value(u),db};
}

RenewalBatchSession::Ticket RenewalBatchSession::submitAsync(const RenewalBatchRequests& r) {
    return submitAsyncImpl<true,false>(r,nullptr);
}

RenewalBatchSession::Ticket RenewalBatchSession::submitWavefrontAsync(
    const RenewalBatchRequests& r, const WavefrontFeatures& features) {
    const auto capacity = static_cast<std::size_t>(impl_->capacity);
    if (features.initial.size() != capacity || features.segment.size() != capacity ||
        features.mixture.size() != capacity)
        throw std::invalid_argument("Renewal wavefront feature capacity mismatch");
#ifndef NDEBUG
    if (pendingCount()) throw std::logic_error("Renewal wavefront must collect before submitting");
    return submitAsyncImpl<true,true>(r,&features);
#else
    return submitAsyncImpl<false,true>(r,&features);
#endif
}

template<bool ValidateSlots, bool Precomputed>
RenewalBatchSession::Ticket RenewalBatchSession::submitAsyncImpl(
    const RenewalBatchRequests& r, const WavefrontFeatures* features) {
    std::size_t batch = impl_->pending.size();
    // At most four entries. Trusted submissions are exclusive even if a future
    // caller mixes entry points; they never rely on per-slot busy flags.
    for (std::size_t i = 0; i < impl_->pending.size(); ++i) {
        if (impl_->pending[i].ticket) {
            if constexpr (!ValidateSlots) {
                throw std::logic_error("Renewal wavefront must collect before submitting");
            } else if (!impl_->pending[i].checkedSlots) {
                throw std::logic_error("Renewal wavefront must collect before submitting");
            }
        } else if (batch == impl_->pending.size()) batch = i;
    }
    if (batch == impl_->pending.size()) throw std::logic_error("Renewal in-flight batch capacity exhausted");
    const auto ni = r.initializeSlots.size(), ns = r.segmentSlots.size(), nm = r.mixtureSlots.size();
    if (ni != r.firstSegments.size() || ni != r.starts.size() || ns != r.segments.size() ||
        nm != r.coordinates.size()) throw std::invalid_argument("Renewal request size mismatch");
    if constexpr (ValidateSlots) {
        auto& seen = impl_->seen;
        std::fill(seen.begin(),seen.end(),0);
        const auto validate = [&](const std::vector<int>& ids, unsigned char bit) {
            for (const int id : ids) {
                if (id < 0 || id >= impl_->capacity || impl_->busy[id] || (seen[id]&bit))
                    throw std::invalid_argument("invalid, duplicate or in-flight Renewal slot");
                if ((bit == 2 && !impl_->status[id] && !(seen[id]&1)) ||
                    (bit == 4 && (impl_->status[id] < 2 || seen[id])))
                    throw std::invalid_argument("uninitialized or conflicting Renewal request");
                seen[id] |= bit;
            }
        };
        validate(r.initializeSlots,1); validate(r.segmentSlots,2); validate(r.mixtureSlots,4);
    }
    auto& pending = impl_->pending[batch];
    pending.clear();
    double* const input = impl_->engine->prepareInput(batch,5*ni+6*ns+4*nm);
    std::size_t offset = 0;
    const auto append = [&](double value) { input[offset++] = value; };
    for (std::size_t i = 0; i < ni; ++i) {
        std::array<double,4> known;
        if constexpr (Precomputed) {
            known = features->initial[r.initializeSlots[i]];
#ifndef NDEBUG
            if (known != initialFeatures(r.firstSegments[i],r.starts[i]))
                throw std::logic_error("stale Renewal initial features");
#endif
        } else {
            known = initialFeatures(r.firstSegments[i],r.starts[i]);
        }
        append(r.initializeSlots[i]);
        for (double x : known) append(finiteInput(x));
    }
    for (std::size_t i = 0; i < ns; ++i) {
        const auto& segment = r.segments[i];
        if (!(segment.end > segment.begin)) throw std::invalid_argument("invalid Renewal batch segment");
        std::array<double,5> values;
        if constexpr (Precomputed) {
            values = features->segment[r.segmentSlots[i]];
#ifndef NDEBUG
            if (values != segment.features()) throw std::logic_error("stale Renewal segment features");
#endif
        } else {
            values = segment.features();
        }
        append(r.segmentSlots[i]);
        for (double x : values) append(finiteInput(x));
    }
    auto& derivatives = pending.derivatives;
    derivatives.reserve(nm);
    for (std::size_t i = 0; i < nm; ++i) {
        const double u = r.coordinates[i];
        if (!(u >= 0 && u <= 1)) throw std::invalid_argument("invalid Renewal batch mixture coordinate");
        std::array<double,2> values;
        if constexpr (Precomputed) {
            values = features->mixture[r.mixtureSlots[i]];
#ifndef NDEBUG
            if (values != mixtureFeatures(impl_->last[r.mixtureSlots[i]],u))
                throw std::logic_error("stale Renewal mixture features");
#endif
        } else {
            values = mixtureFeatures(impl_->last[r.mixtureSlots[i]],u);
        }
        derivatives.push_back(values[1]);
        append(r.mixtureSlots[i]); append(u);
        for (double x : values) append(finiteInput(x));
    }
    pending.checkedSlots = ValidateSlots;
    pending.initialSlots = r.initializeSlots; pending.segmentSlots = r.segmentSlots;
    pending.mixtureSlots = r.mixtureSlots; pending.segments = r.segments;
    if constexpr (ValidateSlots)
        for (int id = 0; id < impl_->capacity; ++id) if (impl_->seen[id]) pending.touched.push_back(id);
    const Ticket ticket = pending.ticket = impl_->nextTicket++;
    if constexpr (ValidateSlots)
        for (int id : pending.touched) impl_->busy[id] = 1;
    try {
        impl_->engine->submit(batch,ni,ns,nm);
    } catch (...) {
        impl_->release(pending,true); throw;
    }
    return ticket;
}

bool RenewalBatchSession::isReady(Ticket ticket) const {
    return impl_->engine->isReady(impl_->find(ticket));
}
std::size_t RenewalBatchSession::pendingCount() const {
    return static_cast<std::size_t>(std::count_if(impl_->pending.begin(),impl_->pending.end(),
        [](const auto& p) { return p.ticket != 0; }));
}
std::optional<RenewalBatchResults> RenewalBatchSession::tryCollect(Ticket ticket) {
    if (!isReady(ticket)) return std::nullopt;
    return collect(ticket);
}
RenewalBatchResults RenewalBatchSession::collect(Ticket ticket) {
    RenewalBatchResults result;
    collectInto(ticket,result);
    return result;
}
void RenewalBatchSession::collectInto(Ticket ticket, RenewalBatchResults& result) {
    const auto batch = impl_->find(ticket);
    auto& p = impl_->pending[batch];
    try {
    const auto raw = impl_->engine->collect(batch);
    const auto ni = p.initialSlots.size(), ns = p.segmentSlots.size(), nm = p.mixtureSlots.size();
    const int n = impl_->weights->components;
    if (raw.size() != ni+5*ns+4*n*nm) throw std::runtime_error("invalid Renewal backend result size");
    // Validity flags share the same readback as hazard and mixture results.
    // Invalidate touched slots on a numerical failure; callers must reinitialize.
    for (int id : p.initialSlots) impl_->status[id] = 0;
    for (int id : p.segmentSlots) impl_->status[id] = 0;
    for (std::size_t i = 0; i < raw.size(); ++i)
        if (!std::isfinite(raw[i])) throw std::runtime_error("nonfinite Renewal batch output");
    for (std::size_t i = 0; i < ni; ++i)
        if (raw[i] != 1.f) throw std::runtime_error("nonfinite LibTorch Renewal initial state");
    // The private wavefront destination retains high-water storage, including
    // nested mixture vectors. Its requests define the active prefixes. Public
    // collect starts with an empty result and therefore returns exact sizes.
    if (result.rates.size() < ns) result.rates.resize(ns);
    if (result.mixtures.size() < nm) result.mixtures.resize(nm);
    for (std::size_t i = 0; i < ns; ++i) {
        if (raw[ni+5*i+4] != 1.f) throw std::runtime_error("nonfinite LibTorch Renewal recurrent state");
        for (int j = 0; j < 4; ++j) result.rates[i][j] = raw[ni+5*i+j];
    }
    for (std::size_t i = 0; i < nm; ++i) {
        const float* row = raw.data()+ni+5*ns+i*4*n;
        const float maximum = *std::max_element(row,row+n);
        double total = 0;
        auto& mixture = result.mixtures[i];
        mixture.weights.resize(n); mixture.means.resize(n); mixture.scales.resize(n);
        for (int j = 0; j < n; ++j) {
            const double w = std::exp(static_cast<double>(row[j]-maximum));
            mixture.weights[j] = w; total += w;
            mixture.means[j] = -p.derivatives[i]+row[n+j];
            mixture.scales[j] = row[3*n+j]+impl_->weights->sigmaFloor;
        }
        for (auto& w : mixture.weights) w /= total;
    }
    for (int id : p.initialSlots) impl_->status[id] = 1;
    for (std::size_t i = 0; i < ns; ++i) {
        impl_->last[p.segmentSlots[i]] = p.segments[i]; impl_->status[p.segmentSlots[i]] = 2;
    }
    impl_->release(p,false);
    } catch (...) {
        impl_->release(p,true); throw;
    }
}
RenewalBatchResults RenewalBatchSession::submit(const RenewalBatchRequests& requests) {
    return collect(submitAsync(requests));
}

void RenewalBatchSession::initialize(const std::vector<int>& slots, const std::vector<RayMeanSegment>& first,
                                    const std::vector<RayStartCondition>& starts) {
    RenewalBatchRequests r;
    r.initializeSlots = slots; r.firstSegments = first; r.starts = starts;
    submit(r);
}
std::vector<std::array<double,4>> RenewalBatchSession::evaluate(const std::vector<int>& slots,
                                                              const std::vector<RayMeanSegment>& segments) {
    RenewalBatchRequests r; r.segmentSlots = slots; r.segments = segments;
    return submit(r).rates;
}
std::vector<RenewalSpeedMixture> RenewalBatchSession::mixture(const std::vector<int>& slots,
                                                            const std::vector<double>& u) {
    RenewalBatchRequests r; r.mixtureSlots = slots; r.coordinates = u;
    return submit(r).mixtures;
}
} // namespace mf

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
    std::vector<double> input;
    std::vector<unsigned char> seen;
    void validateSlots(const std::vector<int>& slots, std::size_t count, int required) {
        if (slots.size() != count || slots.size() > static_cast<std::size_t>(capacity))
            throw std::invalid_argument("Renewal batch size mismatch");
        std::fill(seen.begin(), seen.end(), 0);
        for (int slot : slots) {
            if (slot < 0 || slot >= capacity || seen[slot] || status[slot] < required)
                throw std::invalid_argument("invalid, duplicate or uninitialized Renewal slot");
            seen[slot] = 1;
        }
    }
};

RenewalBatchSession::RenewalBatchSession(const RenewalHazardModel& model, int capacity,
                                       const std::string& backend) : impl_(std::make_unique<Impl>()) {
    if (capacity < 1 || capacity > 65536) throw std::invalid_argument("Renewal batch capacity must be 1..65536");
    impl_->capacity = capacity;
    impl_->name = resolveRenewalBackend(backend);
    if (impl_->name == "scalar") throw std::invalid_argument("scalar backend has no batch session");
    impl_->weights = model.batchWeights();
    impl_->status.resize(capacity, 0); impl_->last.resize(capacity); impl_->seen.resize(capacity);
#ifdef MACROFACET_HAS_TORCH
    impl_->engine = makeRenewalTorchBackend(*impl_->weights, capacity, impl_->name == "torch_cuda");
#endif
}
RenewalBatchSession::~RenewalBatchSession() = default;
const std::string& RenewalBatchSession::backend() const { return impl_->name; }

void RenewalBatchSession::initialize(const std::vector<int>& slots, const std::vector<RayMeanSegment>& first,
                                    const std::vector<RayStartCondition>& starts) {
    impl_->validateSlots(slots, first.size(), 0);
    if (starts.size() != slots.size()) throw std::invalid_argument("Renewal batch start count mismatch");
    if (slots.empty()) return;
    auto& input = impl_->input;
    input.clear(); input.reserve(4*slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!(first[i].end > first[i].begin)) throw std::invalid_argument("invalid Renewal initial segment");
        const double b = first[i].value(first[i].begin);
        std::array<double,4> known{0,b,0,0};
        if (starts[i].mode == RayStartMode::SurfaceOutward) {
            if (!(starts[i].outwardDerivative > 0) || !std::isfinite(starts[i].outwardDerivative))
                throw std::invalid_argument("Renewal surface start requires positive finite derivative");
            known = {1,b,-b,starts[i].outwardDerivative-first[i].derivative(first[i].begin)};
        } else if (starts[i].mode != RayStartMode::PositiveExterior) {
            throw std::invalid_argument("invalid Renewal start mode");
        }
        for (double x : known) input.push_back(finiteInput(x));
    }
    impl_->engine->initialize(slots, input);
    for (int slot : slots) impl_->status[slot] = 1;
}

std::vector<std::array<double,4>> RenewalBatchSession::evaluate(const std::vector<int>& slots,
                                                              const std::vector<RayMeanSegment>& segments) {
    impl_->validateSlots(slots, segments.size(), 1);
    if (slots.empty()) return {};
    auto& input = impl_->input;
    input.clear(); input.reserve(5*slots.size());
    for (const auto& segment : segments) {
        if (!(segment.end > segment.begin)) throw std::invalid_argument("invalid Renewal batch segment");
        const auto f = segment.features();
        for (double x : f) input.push_back(finiteInput(x));
    }
    const auto raw = impl_->engine->evaluate(slots, input);
    std::vector<std::array<double,4>> result(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            const float value = raw[4*i+j]; // Already softplus-transformed by the backend.
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite Renewal batch hazard");
            result[i][j] = value;
        }
        impl_->last[slots[i]] = segments[i]; impl_->status[slots[i]] = 2;
    }
    return result;
}

std::vector<RenewalSpeedMixture> RenewalBatchSession::mixture(const std::vector<int>& slots,
                                                            const std::vector<double>& u) {
    impl_->validateSlots(slots, u.size(), 2);
    if (slots.empty()) return {};
    auto& input = impl_->input;
    input.clear(); input.reserve(3*slots.size());
    std::vector<double> derivatives;
    derivatives.reserve(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!(u[i] >= 0 && u[i] <= 1)) throw std::invalid_argument("invalid Renewal batch mixture coordinate");
        const auto& segment = impl_->last[slots[i]];
        const double db = segment.polynomial.derivative(u[i])/(segment.end-segment.begin);
        derivatives.push_back(db);
        input.push_back(finiteInput(u[i]));
        input.push_back(finiteInput(segment.polynomial.value(u[i])));
        input.push_back(finiteInput(db));
    }
    const auto raw = impl_->engine->mixture(slots, input);
    const int n = impl_->weights->components;
    std::vector<RenewalSpeedMixture> result(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const float* p = raw.data()+i*3*n;
        for (int j = 0; j < 3*n; ++j)
            if (!std::isfinite(p[j])) throw std::runtime_error("nonfinite Renewal batch mixture");
        const float maximum = *std::max_element(p, p+n);
        double total = 0;
        auto& mixture = result[i];
        for (int j = 0; j < n; ++j) {
            const double w = std::exp(static_cast<double>(p[j]-maximum));
            mixture.weights.push_back(w); total += w;
            mixture.means.push_back(-derivatives[i]+p[n+j]);
            mixture.scales.push_back(p[2*n+j]+impl_->weights->sigmaFloor);
        }
        for (auto& w : mixture.weights) w /= total;
    }
    return result;
}
} // namespace mf

#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace mf {
struct RenewalDenseWeights {
    int input = 0, output = 0;
    std::vector<float> weight, bias; // output x input, row major
};
struct RenewalNetworkWeights {
    int hidden = 0, embedding = 0, components = 0;
    float sigmaFloor = 0;
    RenewalDenseWeights encoder0, encoder2, initial0, initial2, initial4;
    RenewalDenseWeights hazard0, hazard2, hazard4, mixture0, mixture2, mixture4;
    RenewalDenseWeights gruInput, gruHidden;
};

// Borrowed host output. The session decodes it before reusing this batch's
// buffers; no view may escape into a public result or a CPU worker task.
struct RenewalBatchOutputView {
    const float* values = nullptr;
    std::size_t count = 0;
    const float* data() const { return values; }
    std::size_t size() const { return count; }
    float operator[](std::size_t index) const { return values[index]; }
};

// Packed inputs/outputs are sample-major (sample x feature row-major).
// State and cached entering context remain in the backend's memory.
// Inputs are raw doubles: the backend applies asinh BEFORE casting to float32.
// initialize transforms all four columns; evaluate transforms columns 0..3
// (column 4 already contains log dx); mixture transforms columns 1..2, not u.
// evaluate returns positive hazard rates. mixture returns weight logits,
// residual means and softplus scales (the session adds sigmaFloor).
class RenewalBatchBackend {
public:
    virtual ~RenewalBatchBackend() = default;
    // Contiguous double rows: [slot,known x4], [slot,features x5],
    // [slot,u,b,db]. Integer slot IDs are exactly representable as doubles.
    // Return: initialization validity x I, [rates x4,state validity] x S,
    // [raw mixture parameters x3K,softplus scales xK] x M.
    // Only prepare an unused/collected batch. The writable host pointer remains
    // valid until its next prepareInput; after submit, don't write until collect.
    // Preparing/packing inputs must not mutate recurrent device state.
    virtual double* prepareInput(std::size_t batch, std::size_t count) = 0;
    virtual void submit(std::size_t batch,
        std::size_t initialCount, std::size_t segmentCount, std::size_t mixtureCount) = 0;
    virtual bool isReady(std::size_t batch) const = 0;
    // Waits for completion. View expires on this batch's next prepare/submit,
    // or backend destruction. Other batches own independent host buffers.
    virtual RenewalBatchOutputView collect(std::size_t batch) = 0;
};
#ifdef MACROFACET_HAS_TORCH
std::unique_ptr<RenewalBatchBackend> makeRenewalTorchBackend(
    const RenewalNetworkWeights&, int capacity, bool cuda, int maximumInFlight);
bool queryRenewalTorchCuda(std::string* reason);
#endif
} // namespace mf

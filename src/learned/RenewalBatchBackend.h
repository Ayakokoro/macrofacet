#pragma once
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
    virtual void initialize(const std::vector<int>& slots, const std::vector<double>& known) = 0;
    virtual std::vector<float> evaluate(const std::vector<int>& slots, const std::vector<double>& features) = 0;
    virtual std::vector<float> mixture(const std::vector<int>& slots, const std::vector<double>& query) = 0;
};
#ifdef MACROFACET_HAS_TORCH
std::unique_ptr<RenewalBatchBackend> makeRenewalTorchBackend(
    const RenewalNetworkWeights&, int capacity, bool cuda);
bool queryRenewalTorchCuda(std::string* reason);
#endif
} // namespace mf

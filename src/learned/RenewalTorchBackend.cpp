#include "RenewalBatchBackend.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {
struct Dense {
    at::Tensor weight, bias;
    Dense(const RenewalDenseWeights& layer, const at::Device& device) {
        weight = at::from_blob(const_cast<float*>(layer.weight.data()), {layer.output,layer.input},
                              at::kFloat).clone().to(device);
        bias = at::from_blob(const_cast<float*>(layer.bias.data()), {layer.output},
                            at::kFloat).clone().to(device);
    }
    at::Tensor operator()(const at::Tensor& input) const { return at::linear(input, weight, bias); }
};
struct Mlp {
    Dense first, second, last;
    Mlp(const RenewalDenseWeights& a, const RenewalDenseWeights& b, const RenewalDenseWeights& c,
        const at::Device& device) : first(a,device), second(b,device), last(c,device) {}
    at::Tensor operator()(const at::Tensor& input) const {
        return last(at::silu(second(at::silu(first(input)))));
    }
};

class TorchBackend final : public RenewalBatchBackend {
    at::Device device_;
    Dense encoder0_, encoder2_, gruInput_, gruHidden_;
    Mlp initial_, hazard_, mixture_;
    at::Tensor states_, enteringContext_;
    int components_;

    at::Tensor indices(const std::vector<int>& slots) const {
        // Own the CPU storage too: the source vector does not outlive this call.
        return at::from_blob(const_cast<int*>(slots.data()), {static_cast<int64_t>(slots.size())}, at::kInt)
            .to(at::TensorOptions().dtype(at::kLong).device(device_), false, true);
    }
    at::Tensor input(const std::vector<double>& values, int width) const {
        return at::from_blob(const_cast<double*>(values.data()),
            {static_cast<int64_t>(values.size()/width),width}, at::kDouble)
            .to(at::TensorOptions().dtype(at::kDouble).device(device_), false, true);
    }
    static at::Tensor transformInput(at::Tensor values, int first, int count) {
        // input() owns this storage on both CPU and CUDA. Preserve the original
        // double asinh -> float conversion, including values beyond FLT_MAX.
        values.narrow(1,first,count).asinh_();
        return values.to(at::kFloat);
    }
    static std::vector<float> download(const at::Tensor& result) {
        const auto host = result.to(at::kCPU).contiguous();
        const float* values = host.const_data_ptr<float>();
        return {values, values+host.numel()};
    }
public:
    TorchBackend(const RenewalNetworkWeights& w, int capacity, bool cuda)
        : device_(cuda ? at::kCUDA : at::kCPU), encoder0_(w.encoder0,device_), encoder2_(w.encoder2,device_),
          gruInput_(w.gruInput,device_), gruHidden_(w.gruHidden,device_),
          initial_(w.initial0,w.initial2,w.initial4,device_),
          hazard_(w.hazard0,w.hazard2,w.hazard4,device_),
          mixture_(w.mixture0,w.mixture2,w.mixture4,device_), components_(w.components) {
        states_ = at::zeros({capacity,w.hidden}, at::TensorOptions().dtype(at::kFloat).device(device_));
        enteringContext_ = at::zeros({capacity,w.hidden+w.embedding}, states_.options());
    }
    void initialize(const std::vector<int>& slots, const std::vector<double>& known) override {
        c10::InferenceMode inference;
        at::NoTF32Guard fullPrecision;
        const auto state = initial_(transformInput(input(known,4),0,4));
        if (!at::isfinite(state).all().item<bool>())
            throw std::runtime_error("nonfinite LibTorch Renewal initial state");
        states_.index_copy_(0, indices(slots), state);
    }
    std::vector<float> evaluate(const std::vector<int>& slots, const std::vector<double>& features) override {
        c10::InferenceMode inference;
        at::NoTF32Guard fullPrecision;
        const auto ids = indices(slots);
        const auto old = states_.index_select(0,ids);
        const auto featuresDevice = input(features,5);
        const auto networkFeatures = transformInput(featuresDevice,0,4);
        const auto embedded = at::silu(encoder2_(at::silu(encoder0_(networkFeatures))));
        const auto context = at::cat({old,embedded},1);
        const auto logits = hazard_(context);
        const auto rates = at::softplus(logits,1,20);
        // LibTorch's fused CUDA GRU cell implements PyTorch's reset-after rule.
        const auto next = at::gru_cell(embedded,old,gruInput_.weight,gruHidden_.weight,
                                      gruInput_.bias,gruHidden_.bias);
        enteringContext_.index_copy_(0,ids,context);
        states_.index_copy_(0,ids,next);
        // Piggyback state validity on the small hazard transfer, not a second sync.
        const auto packedDevice = at::cat({rates,at::isfinite(next).all(1,true).to(at::kFloat)},1);
        const auto packed = download(packedDevice);
        std::vector<float> result(4*slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (packed[5*i+4] != 1.f) throw std::runtime_error("nonfinite LibTorch Renewal recurrent state");
            for (int j = 0; j < 4; ++j) result[4*i+j] = packed[5*i+j];
        }
        return result;
    }
    std::vector<float> mixture(const std::vector<int>& slots, const std::vector<double>& query) override {
        c10::InferenceMode inference;
        at::NoTF32Guard fullPrecision;
        const auto context = enteringContext_.index_select(0,indices(slots));
        const auto parameters = mixture_(at::cat({context,transformInput(input(query,3),1,2)},1));
        const auto scaleLogits = parameters.narrow(1,2*components_,components_);
        const auto scales = at::softplus(scaleLogits,1,20);
        // Retain the raw scale logits in the existing download: softplus can
        // hide -inf as zero. Checking these few values on CPU avoids extra
        // reduction kernels and synchronization for the small hit batches.
        const auto packed = download(at::cat({parameters,scales},1));
        const int width = 3*components_;
        std::vector<float> result(width*slots.size());
        for (std::size_t i = 0; i < slots.size(); ++i) {
            const float* row = packed.data()+4*components_*i;
            for (int j = 0; j < components_; ++j)
                if (!std::isfinite(row[2*components_+j]))
                    throw std::runtime_error("nonfinite LibTorch Renewal mixture scale logits");
            std::copy_n(row,2*components_,result.data()+width*i);
            std::copy_n(row+3*components_,components_,result.data()+width*i+2*components_);
        }
        return result;
    }
};
}

std::unique_ptr<RenewalBatchBackend> makeRenewalTorchBackend(
    const RenewalNetworkWeights& weights, int capacity, bool cuda) {
    c10::InferenceMode inference;
    return std::make_unique<TorchBackend>(weights,capacity,cuda);
}
bool queryRenewalTorchCuda(std::string* reason) {
    try {
        if (torch::cuda::is_available()) return true;
        if (reason) *reason = "LibTorch reports no available CUDA device";
    } catch (const std::exception& e) {
        if (reason) *reason = e.what();
    }
    return false;
}
} // namespace mf

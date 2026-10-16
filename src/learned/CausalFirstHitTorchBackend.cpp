#include "CausalFirstHitBackend.h"
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>

namespace mf {
namespace {
struct Dense {
    at::Tensor weight,bias;
    Dense(const RenewalDenseWeights& w,const at::Device& device) {
        weight=at::from_blob(const_cast<float*>(w.weight.data()),{w.output,w.input},at::kFloat).clone().to(device);
        bias=at::from_blob(const_cast<float*>(w.bias.data()),{w.output},at::kFloat).clone().to(device);
    }
    at::Tensor operator()(const at::Tensor& x) const { return at::linear(x,weight,bias); }
};
class TorchBackend final : public CausalFirstHitBackend {
    at::Device device_;
    Dense initial0_,initial2_,segment0_,segment2_,head0_,head2_,mixture0_,mixture2_,mixture4_,gruInput_,gruHidden_;
    at::Tensor states_;
    int components_;
    at::Tensor inputs(const std::vector<float>& values,int columns) const {
        return at::from_blob(const_cast<float*>(values.data()),{static_cast<int64_t>(values.size()/columns),columns},at::kFloat).to(device_);
    }
    at::Tensor indices(const std::vector<int>& slots) const {
        return at::from_blob(const_cast<int*>(slots.data()),{static_cast<int64_t>(slots.size())},at::kInt).to(device_,at::kLong);
    }
    static std::vector<float> collect(const at::Tensor& tensor) {
        const auto cpu=tensor.to(at::kCPU).contiguous();
        const auto* data=cpu.data_ptr<float>(); return {data,data+cpu.numel()};
    }
public:
    TorchBackend(const CausalNetworkWeights& w,bool cuda):device_(cuda ? at::kCUDA : at::kCPU),
        initial0_(w.initial0,device_),initial2_(w.initial2,device_),segment0_(w.segment0,device_),segment2_(w.segment2,device_),
        head0_(w.head0,device_),head2_(w.head2,device_),mixture0_(w.mixture0,device_),mixture2_(w.mixture2,device_),
        mixture4_(w.mixture4,device_),gruInput_(w.gruInput,device_),gruHidden_(w.gruHidden,device_),components_(w.components) {}
    void initialize(const std::vector<float>& entries) override {
        c10::InferenceMode guard; states_=initial2_(at::silu(initial0_(inputs(entries,2))));
    }
    std::vector<float> correction(const std::vector<int>& slots,const std::vector<float>& left) override {
        c10::InferenceMode guard;
        return collect(head2_(at::silu(head0_(at::cat({states_.index_select(0,indices(slots)),inputs(left,3)},1)))));
    }
    void advance(const std::vector<int>& slots,const std::vector<float>& tokens) override {
        c10::InferenceMode guard; const auto index=indices(slots);
        const auto embedded=at::silu(segment2_(at::silu(segment0_(inputs(tokens,7)))));
        const auto next=at::gru_cell(embedded,states_.index_select(0,index),gruInput_.weight,gruHidden_.weight,gruInput_.bias,gruHidden_.bias);
        states_.index_copy_(0,index,next);
    }
    std::vector<float> mixture(const std::vector<int>& slots,const std::vector<float>& local) override {
        c10::InferenceMode guard;
        const auto raw=mixture4_(at::silu(mixture2_(at::silu(mixture0_(at::cat({states_.index_select(0,indices(slots)),inputs(local,4)},1))))));
        raw.narrow(1,2*components_,components_).copy_(at::softplus(raw.narrow(1,2*components_,components_))+.01f);
        return collect(raw);
    }
};
}
std::unique_ptr<CausalFirstHitBackend> makeCausalTorchBackend(const CausalNetworkWeights& w,bool cuda) {
    c10::InferenceMode guard; return std::make_unique<TorchBackend>(w,cuda);
}
} // namespace mf

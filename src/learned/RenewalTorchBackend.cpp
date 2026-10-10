#include "RenewalBatchBackend.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <optional>
#ifdef MACROFACET_TORCH_CUDA
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
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

    struct Batch {
        at::Tensor hostInput, deviceInput, hostOutput, deviceOutput;
        int64_t inputCount = 0, outputCount = 0;
#ifdef MACROFACET_TORCH_CUDA
        c10::cuda::CUDAEvent completed;
#endif
    };
    std::vector<Batch> batches_;
#ifdef MACROFACET_TORCH_CUDA
    std::optional<c10::cuda::CUDAStream> stream_;
#endif
    static at::Tensor transformInput(at::Tensor values, int first, int count) {
        // Preserve double asinh -> float, including values beyond FLT_MAX.
        values.narrow(1,first,count).asinh_();
        return values.to(at::kFloat);
    }
public:
    TorchBackend(const RenewalNetworkWeights& w, int capacity, bool cuda, int maximumInFlight)
        : device_(cuda ? at::kCUDA : at::kCPU), encoder0_(w.encoder0,device_), encoder2_(w.encoder2,device_),
          gruInput_(w.gruInput,device_), gruHidden_(w.gruHidden,device_),
          initial_(w.initial0,w.initial2,w.initial4,device_),
          hazard_(w.hazard0,w.hazard2,w.hazard4,device_),
          mixture_(w.mixture0,w.mixture2,w.mixture4,device_), components_(w.components), batches_(maximumInFlight) {
        states_ = at::zeros({capacity,w.hidden}, at::TensorOptions().dtype(at::kFloat).device(device_));
        enteringContext_ = at::zeros({capacity,w.hidden+w.embedding}, states_.options());
#ifdef MACROFACET_TORCH_CUDA
        if (cuda) {
            stream_ = c10::cuda::getStreamFromPool(false,states_.get_device());
            c10::cuda::CUDAEvent initialized;
            initialized.record(c10::cuda::getCurrentCUDAStream(states_.get_device()));
            initialized.block(*stream_);
        }
#else
        if (cuda) throw std::runtime_error("LibTorch backend built without CUDA stream support");
#endif
    }
    ~TorchBackend() override {
#ifdef MACROFACET_TORCH_CUDA
        // Drain work before destroying any pinned host or device buffers, also
        // when a caller abandons outstanding tickets or unwinds after an error.
        if (stream_) { try { stream_->synchronize(); } catch (...) {} }
#endif
    }
    double* prepareInput(std::size_t batch, std::size_t requested) override {
        c10::InferenceMode inference;
        auto& frame = batches_.at(batch);
        const auto count = static_cast<int64_t>(requested);
        frame.inputCount = count;
        if (!count) return nullptr;
#ifdef MACROFACET_TORCH_CUDA
        c10::cuda::OptionalCUDAStreamGuard guard;
        if (stream_) guard.reset_stream(*stream_);
#endif
        if (!frame.hostInput.defined() || !frame.deviceInput.defined() ||
            frame.hostInput.numel() < count || frame.deviceInput.numel() < count) {
            const auto allocation = std::max<int64_t>(count,frame.hostInput.defined() ? 2*frame.hostInput.numel() : 4096);
            // Commit both allocations together; failure leaves the old pair usable.
            auto host = at::empty({allocation},at::TensorOptions().dtype(at::kDouble)
                .device(at::kCPU).pinned_memory(device_.is_cuda()));
            auto device = at::empty({allocation},host.options().pinned_memory(false).device(device_));
            frame.hostInput = std::move(host); frame.deviceInput = std::move(device);
        }
        return frame.hostInput.data_ptr<double>();
    }
    void submit(std::size_t batch,
        std::size_t initialCount, std::size_t segmentCount, std::size_t mixtureCount) override {
        c10::InferenceMode inference;
        at::NoTF32Guard fullPrecision;
        auto& frame = batches_.at(batch);
        const auto count = frame.inputCount;
        if (static_cast<std::size_t>(count) != 5*initialCount+6*segmentCount+4*mixtureCount)
            throw std::logic_error("Renewal prepared input size mismatch");
        frame.outputCount = 0;
        if (!count) return;
#ifdef MACROFACET_TORCH_CUDA
        c10::cuda::OptionalCUDAStreamGuard guard;
        if (stream_) guard.reset_stream(*stream_);
#endif
        try {
        // Each in-flight batch owns pinned staging; no buffer is reused until collect.
        const auto deviceInput = frame.deviceInput.narrow(0,0,count);
        deviceInput.copy_(frame.hostInput.narrow(0,0,count),true);
        int64_t offset = 0;
        const auto rows = [&](std::size_t size, int width) {
            auto block = deviceInput.narrow(0,offset,static_cast<int64_t>(size)*width)
                .view({static_cast<int64_t>(size),width});
            offset += static_cast<int64_t>(size)*width;
            return block;
        };
        std::vector<at::Tensor> outputs;
        if (initialCount) {
            const auto block = rows(initialCount,5);
            const auto ids = block.select(1,0).to(at::kLong);
            const auto state = initial_(transformInput(block.narrow(1,1,4),0,4));
            states_.index_copy_(0,ids,state);
            outputs.push_back(at::isfinite(state).all(1).to(at::kFloat));
        }
        if (segmentCount) {
            const auto block = rows(segmentCount,6);
            const auto ids = block.select(1,0).to(at::kLong);
            const auto old = states_.index_select(0,ids);
            const auto networkFeatures = transformInput(block.narrow(1,1,5),0,4);
            const auto embedded = at::silu(encoder2_(at::silu(encoder0_(networkFeatures))));
            const auto context = at::cat({old,embedded},1);
            const auto logits = hazard_(context);
            const auto rates = at::softplus(logits,1,20);
            const auto next = at::gru_cell(embedded,old,gruInput_.weight,gruHidden_.weight,
                                          gruInput_.bias,gruHidden_.bias);
            enteringContext_.index_copy_(0,ids,context);
            states_.index_copy_(0,ids,next);
            outputs.push_back(at::cat({rates,at::isfinite(next).all(1,true).to(at::kFloat)},1).flatten());
        }
        if (mixtureCount) {
            const auto block = rows(mixtureCount,4);
            const auto ids = block.select(1,0).to(at::kLong);
            const auto context = enteringContext_.index_select(0,ids);
            const auto parameters = mixture_(at::cat({context,transformInput(block.narrow(1,1,3),1,2)},1));
            const auto scales = at::softplus(parameters.narrow(1,2*components_,components_),1,20);
            // Include raw scale logits: softplus(-inf) alone would hide overflow.
            outputs.push_back(at::cat({parameters,scales},1).flatten());
        }
        frame.deviceOutput = at::cat(outputs,0);
        frame.outputCount = frame.deviceOutput.numel();
        if (!frame.hostOutput.defined() || frame.hostOutput.numel() < frame.outputCount) {
            const auto allocation = std::max<int64_t>(frame.outputCount,
                frame.hostOutput.defined() ? 2*frame.hostOutput.numel() : 4096);
            frame.hostOutput = at::empty({allocation},at::TensorOptions().dtype(at::kFloat)
                .device(at::kCPU).pinned_memory(device_.is_cuda()));
        }
        // Queue D2H into pinned storage. Reading it is legal only after the event.
        frame.hostOutput.narrow(0,0,frame.outputCount).copy_(frame.deviceOutput,true);
#ifdef MACROFACET_TORCH_CUDA
        if (stream_) frame.completed.record(*stream_);
#endif
        } catch (...) {
#ifdef MACROFACET_TORCH_CUDA
            if (stream_) { try { stream_->synchronize(); } catch (...) {} }
#endif
            throw;
        }
    }

    bool isReady(std::size_t batch) const override {
        const auto& frame = batches_.at(batch);
#ifdef MACROFACET_TORCH_CUDA
        if (stream_ && frame.outputCount) return frame.completed.query();
#endif
        return true;
    }
    RenewalBatchOutputView collect(std::size_t batch) override {
        auto& frame = batches_.at(batch);
        if (!frame.outputCount) return {};
#ifdef MACROFACET_TORCH_CUDA
        if (stream_) frame.completed.synchronize();
#endif
        const auto* values = frame.hostOutput.data_ptr<float>();
        return {values,static_cast<std::size_t>(frame.outputCount)};
    }
};
}

std::unique_ptr<RenewalBatchBackend> makeRenewalTorchBackend(
    const RenewalNetworkWeights& weights, int capacity, bool cuda, int maximumInFlight) {
    c10::InferenceMode inference;
    return std::make_unique<TorchBackend>(weights,capacity,cuda,maximumInFlight);
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

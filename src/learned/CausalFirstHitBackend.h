#pragma once
#include "RenewalBatchBackend.h"

namespace mf {
struct CausalNetworkWeights {
    int hidden=0, embedding=0, components=0;
    RenewalDenseWeights initial0,initial2,segment0,segment2,head0,head2;
    RenewalDenseWeights mixture0,mixture2,mixture4,gruInput,gruHidden;
};
// Inputs already transformed in double precision then cast to float32.
// Stable slot IDs index persistent device state, even after active compaction.
class CausalFirstHitBackend {
public:
    virtual ~CausalFirstHitBackend()=default;
    virtual void initialize(const std::vector<float>& entries)=0;
    virtual std::vector<float> correction(const std::vector<int>& slots,const std::vector<float>& left)=0;
    virtual void advance(const std::vector<int>& slots,const std::vector<float>& tokens)=0;
    // logits, residual means, and (softplus(raw scale)+.01), computed in float32.
    virtual std::vector<float> mixture(const std::vector<int>& slots,const std::vector<float>& local)=0;
};
#ifdef MACROFACET_HAS_TORCH
std::unique_ptr<CausalFirstHitBackend> makeCausalTorchBackend(const CausalNetworkWeights&,bool cuda);
#endif
} // namespace mf

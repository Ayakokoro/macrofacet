#include "macrofacet/learned/CausalFirstHitModel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include "CausalModelInternal.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace mf {
using namespace causal_detail;

struct CausalFirstHitModel::Impl { CausalNetworkWeights weights; double step=0,fine=0; };
CausalFirstHitModel CausalFirstHitModel::load(const std::filesystem::path& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot open causal model "+path.string());
    Json r; f>>r;
    if (r.at("format")!="macrofacet.causal_first_hit" || r.at("version")!=1 ||
        r.at("event")!="unconditioned_first_zero" || r.at("profile_mode")!="cubic" ||
        r.at("kernel")!="squared_exponential" || r.at("activation_dtype")!="float32" ||
        r.at("distance_convention")!="q=distance_from_entry/ell" || r.at("mean_convention")!="b=m/sigma" ||
        r.at("gradient_convention")!="signed_derivative=-initial_sign*sigma/ell*W" ||
        r.at("state_convention")!="entering_fixed_q_interval_left_limit_gradient")
        throw std::invalid_argument("unsupported causal first-hit contract");
    const auto& c=r.at("model_config"); const auto& g=r.at("gradient_config");
    auto p=std::make_shared<Impl>(); auto& w=p->weights;
    w.hidden=c.at("hidden"); w.embedding=c.at("embedding"); w.components=g.at("size");
    const int width=c.at("width"),gw=g.at("width"),h=w.hidden,e=w.embedding;
    p->step=c.at("step"); p->fine=c.at("integration_step");
    if (!(p->step>0 && p->step<=1 && p->fine>0 && p->fine<=p->step) || c.at("quadrature_order")!=8 ||
        c.at("history")!=true || g.at("history")!=true || g.at("kind")!="mixture" || g.at("hidden")!=h ||
        h<1 || h>1024 || e<1 || e>1024 || w.components<1 || w.components>1024)
        throw std::invalid_argument("invalid causal model configuration");
    const auto& b=r.at("backbone"); const auto& m=r.at("gradient");
    w.initial0=dense(b,"initial.0",2,h); w.initial2=dense(b,"initial.2",h,h);
    w.segment0=dense(b,"segment.0",7,e); w.segment2=dense(b,"segment.2",e,e);
    w.head0=dense(b,"head.0",h+3,width); w.head2=dense(b,"head.2",width,1);
    w.gruInput=dense(b,"gru",e,3*h,".weight_ih_l0",".bias_ih_l0");
    w.gruHidden=dense(b,"gru",h,3*h,".weight_hh_l0",".bias_hh_l0");
    w.mixture0=dense(m,"net.0",h+4,gw); w.mixture2=dense(m,"net.2",gw,gw); w.mixture4=dense(m,"net.4",gw,3*w.components);
    return CausalFirstHitModel(std::move(p));
}
double CausalFirstHitModel::step() const { return impl_->step; }
CausalFirstHitUniforms causalFirstHitUniforms(Random& rng) {
    return {rng.openUniform01(),rng.openUniform01(),rng.openUniform01(),rng.openUniform01()};
}
struct CausalFirstHitSession::Impl {
    std::shared_ptr<const CausalFirstHitModel::Impl> model;
    std::unique_ptr<CausalFirstHitBackend> backend;
    std::unique_ptr<RenewalCpuExecutor> executor;
};
void CausalFirstHitSession::validateBackend(const std::string& backend) {
    if (backend=="scalar") return;
    if (backend!="torch_cpu" && backend!="torch_cuda") throw std::invalid_argument("unknown causal backend: "+backend);
#ifdef MACROFACET_HAS_TORCH
    std::string reason;
    if (backend=="torch_cuda" && !queryRenewalTorchCuda(&reason)) throw std::runtime_error("causal CUDA unavailable: "+reason);
#else
    throw std::runtime_error("causal batching requires MACROFACET_ENABLE_TORCH");
#endif
}
CausalFirstHitSession::CausalFirstHitSession(const CausalFirstHitModel& model,const std::string& backend,int cpuThreads)
    : impl_(std::make_unique<Impl>()) {
    validateBackend(backend); impl_->model=model.impl_;
    if (cpuThreads<0) throw std::invalid_argument("invalid causal CPU thread count");
    if (cpuThreads!=1) impl_->executor=std::make_unique<RenewalCpuExecutor>(cpuThreads,65536);
    if (backend=="scalar") impl_->backend=std::make_unique<ScalarBackend>(model.impl_->weights);
#ifdef MACROFACET_HAS_TORCH
    else impl_->backend=makeCausalTorchBackend(model.impl_->weights,backend=="torch_cuda");
#endif
}
CausalFirstHitSession::~CausalFirstHitSession()=default;
std::vector<FirstHitSESample> CausalFirstHitSession::sample(const std::vector<const RayMeanProfile*>& profiles,
    const std::vector<CausalFirstHitUniforms>& uniforms,double tolerance) {
    if (profiles.size()!=uniforms.size() || profiles.size()>65536 || !(tolerance>0) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid causal batch or root tolerance");
    const auto& model=*impl_->model; auto& backend=*impl_->backend;
    const std::size_t count=profiles.size();
    std::vector<FirstHitSESample> result(count);
    if (!count) return result;
    std::vector<float> entries;
    std::vector<int> active; std::vector<double> target(count),hitLeft(count);
    for (std::size_t i=0;i<count;++i) {
        if (!profiles[i]) throw std::invalid_argument("null causal profile");
        validateProfile(*profiles[i]);
        for (double u:uniforms[i]) if (!(u>0 && u<1)) throw std::invalid_argument("causal uniforms must be open");
        auto& r=result[i]; r.q=profiles[i]->maximumX();
        r.initialSign=std::log(uniforms[i][0])<normalLogCdf(profiles[i]->segments().front().value(0)) ? 1 : -1;
        appendEntry(entries,*profiles[i],r.initialSign); active.push_back(static_cast<int>(i));
        target[i]=-std::log(uniforms[i][1]);
    }
    backend.initialize(entries);
    for (std::size_t cell=0;!active.empty();++cell) {
        const double a=cell*model.step;
        std::vector<float> left;
        for (int slot:active) appendLeft(left,*profiles[slot],result[slot].initialSign,a);
        const auto delta=backend.correction(active,left);
        std::vector<char> survives(active.size(),0);
        std::vector<float> completedTokens(active.size()*7);
        const auto integrateRows=[&](std::size_t first,std::size_t last) {
        for (std::size_t row=first;row<last;++row) {
            const int slot=active[row],sign=result[slot].initialSign; const auto& p=*profiles[slot];
            const double b=std::min((cell+1)*model.step,p.maximumX()),factor=std::exp(delta[row]);
            if (!(factor>0) || !std::isfinite(factor)) throw std::runtime_error("nonfinite causal hazard correction");
            const auto parts=fragments(p,sign,a,b,model.fine);
            const double mass=total(parts)*factor;
            if (!std::isfinite(mass)) throw std::runtime_error("nonfinite causal integrated hazard");
            if (mass>0 && target[slot]<=mass) {
                double residual=target[slot]/factor;
                const Fragment* selected=&parts.back();
                for (const auto& part:parts) {
                    if ((residual<=part.mass && part.mass>0) || &part==&parts.back()) { selected=&part; break; }
                    residual-=part.mass;
                }
                residual=std::clamp(residual,0.,selected->mass);
                double lo=selected->a,hi=selected->b;
                for (int iteration=0;iteration<80 && hi-lo>tolerance;++iteration) {
                    const double mid=(lo+hi)/2;
                    if (integral(p,sign,selected->a,mid)<residual) lo=mid; else hi=mid;
                }
                result[slot].hit=true; result[slot].q=(lo+hi)/2; hitLeft[slot]=a;
            } else {
                target[slot]-=mass;
                if (b<p.maximumX()) {
                    survives[row]=1; std::vector<float> token; appendToken(token,p,sign,a,b);
                    std::copy(token.begin(),token.end(),completedTokens.begin()+7*row);
                }
            }
        }
        };
        if (impl_->executor && active.size()>=RenewalCpuExecutor::parallelThreshold)
            impl_->executor->forRanges(active.size(),integrateRows);
        else integrateRows(0,active.size());
        std::vector<int> survivors; std::vector<float> tokens;
        for (std::size_t row=0;row<active.size();++row) if (survives[row]) {
            survivors.push_back(active[row]);
            tokens.insert(tokens.end(),completedTokens.begin()+7*row,completedTokens.begin()+7*(row+1));
        }
        if (!survivors.empty()) backend.advance(survivors,tokens);
        active=std::move(survivors);
    }
    std::vector<int> hits; std::vector<float> local;
    for (std::size_t slot=0;slot<count;++slot) if (result[slot].hit) {
        hits.push_back(static_cast<int>(slot));
        appendLocal(local,*profiles[slot],result[slot].initialSign,result[slot].q,hitLeft[slot],model.step);
    }
    if (!hits.empty()) {
        const auto raw=backend.mixture(hits,local);
        for (std::size_t row=0;row<hits.size();++row) {
            const int slot=hits[row]; auto& r=result[slot];
            const auto m=decode(raw.data()+row*3*model.weights.components,model.weights.components,
                r.initialSign*segmentAt(*profiles[slot],r.q).derivative(r.q));
            r.speed=sampleSpeed(m,uniforms[slot][2],uniforms[slot][3]);
            if (!(r.speed>0) || !std::isfinite(r.speed)) throw std::runtime_error("invalid causal crossing speed");
        }
    }
    return result;
}
FirstHitSESample CausalFirstHitModel::sample(const RayMeanProfile& p,const CausalFirstHitUniforms& u,double tolerance) const {
    CausalFirstHitSession session(*this); return session.sample({&p},{u},tolerance)[0];
}
std::vector<FirstHitSEPrediction> CausalFirstHitModel::query(const RayMeanProfile& p,const std::vector<double>& q) const {
    validateProfile(p);
    for (double x:q) if (!(x>=0 && x<=p.maximumX()) || !std::isfinite(x)) throw std::invalid_argument("causal query outside profile");
    const auto& m=*impl_; const int count=static_cast<int>(std::ceil(p.maximumX()/m.step));
    std::vector<FirstHitSEPrediction> out(q.size());
    for (int branch=0;branch<2;++branch) {
        const int sign=branch==0 ? 1 : -1;
        ScalarBackend backend(m.weights); std::vector<float> entry; appendEntry(entry,p,sign); backend.initialize(entry);
        std::vector<double> prefix{0},deltas;
        for (int cell=0;cell<count;++cell) {
            const double a=cell*m.step,b=std::min((cell+1)*m.step,p.maximumX());
            std::vector<float> left; appendLeft(left,p,sign,a);
            const double delta=backend.correction({0},left)[0]; deltas.push_back(delta);
            prefix.push_back(prefix.back()+std::exp(delta)*total(fragments(p,sign,a,b,m.fine)));
            std::vector<int> slots; std::vector<std::size_t> indices; std::vector<float> local;
            for (std::size_t j=0;j<q.size();++j) {
                const int gradientCell=std::clamp(static_cast<int>(std::ceil(q[j]/m.step))-1,0,count-1);
                if (gradientCell==cell) { slots.push_back(0); indices.push_back(j); appendLocal(local,p,sign,q[j],a,m.step); }
            }
            if (!slots.empty()) {
                const auto raw=backend.mixture(slots,local);
                for (std::size_t i=0;i<indices.size();++i) {
                    const auto j=indices[i]; out[j].mixture[branch]=decode(raw.data()+i*3*m.weights.components,
                        m.weights.components,sign*segmentAt(p,q[j]).derivative(q[j]));
                }
            }
            if (cell+1<count) { std::vector<float> tokens; appendToken(tokens,p,sign,a,b); backend.advance({0},tokens); }
        }
        for (std::size_t j=0;j<q.size();++j) {
            const int cell=std::min(static_cast<int>(std::floor(q[j]/m.step)),count-1);
            out[j].q=q[j];
            out[j].cumulative[branch]=prefix[cell]+std::exp(deltas[cell])*total(fragments(p,sign,cell*m.step,q[j],m.fine));
            const auto& s=segmentAt(p,q[j]); out[j].logHazard[branch]=logLocalRate(sign*s.value(q[j]),sign*s.derivative(q[j]))+deltas[cell];
        }
    }
    const double b0=p.segments().front().value(0);
    for (auto& row:out) {
        const double a=normalLogCdf(b0)-row.cumulative[0],b=normalLogCdf(-b0)-row.cumulative[1];
        row.logTransmittance=logAdd(a,b); row.logDensity=logAdd(a+row.logHazard[0],b+row.logHazard[1]);
        row.hitBranchProbability={std::exp(a+row.logHazard[0]-row.logDensity),std::exp(b+row.logHazard[1]-row.logDensity)};
    }
    return out;
}
} // namespace mf

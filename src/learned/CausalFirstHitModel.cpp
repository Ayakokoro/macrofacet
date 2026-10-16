#include "macrofacet/learned/CausalFirstHitModel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include "CausalFirstHitBackend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace mf {
namespace {
using Json=nlohmann::json;
using Vec=Eigen::VectorXf;
using Mat=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
double logAdd(double a,double b) {
    const double m=std::max(a,b);
    return std::isinf(m) ? m : m+std::log(std::exp(a-m)+std::exp(b-m));
}
double logLocalRate(double b,double d) {
    const double phi=normalLogPdf(d);
    double negativeMean;
    if (d<=0) negativeMean=logAdd(phi,d<0 ? std::log(-d)+normalLogCdf(-d) : -INFINITY);
    else if (d<=26) negativeMean=phi+std::log1p(-d*std::exp(normalLogCdf(-d)-phi));
    else {
        const double v=1/(d*d);
        negativeMean=phi+std::log(v)+std::log(1+v*(-3+v*(15+v*(-105+v*(945+v*(-10395+v*135135))))));
    }
    return (b < -8 ? std::log(normalPdfOverCdf(b)) : normalLogPdf(b)-normalLogCdf(b))+negativeMean;
}
const RayMeanSegment& segmentAt(const RayMeanProfile& p,double q) {
    const auto& s=p.segments();
    auto it=std::lower_bound(s.begin(),s.end(),q,[](const auto& a,double x){return a.end<x;});
    return it==s.end() ? s.back() : *it;
}
void validateProfile(const RayMeanProfile& p) {
    if (p.segments().empty() || !(p.maximumX()>0) || !std::isfinite(p.maximumX()))
        throw std::invalid_argument("invalid causal first-hit profile");
    double value=p.segments().front().value(0),end=0;
    for (const auto& s:p.segments()) {
        if (std::abs(s.begin-end)>1e-10 || !(s.end>s.begin) ||
            !std::isfinite(s.value(s.begin)) || !std::isfinite(s.value(s.end)) ||
            !std::isfinite(s.derivative(s.begin)) || !std::isfinite(s.derivative(s.end)) ||
            std::abs(s.value(s.begin)-value)>1e-8*(1+std::abs(value)))
            throw std::invalid_argument("causal first-hit requires a continuous cubic profile from q=0");
        value=s.value(s.end); end=s.end;
    }
}
// Eight-point Gauss-Legendre. Every integration fragment stays inside a cubic
// and an integration-grid cell; reducing root tolerance never advances the GRU.
double integral(const RayMeanProfile& p,int sign,double a,double b) {
    static constexpr double x[]={.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
    static constexpr double w[]={.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
    const double mid=(a+b)/2,half=(b-a)/2;
    if (!(half>0)) return 0;
    const auto& segment=segmentAt(p,mid);
    double sum=0;
    for (int j=0;j<4;++j) for (int direction:{-1,1}) {
        const double q=mid+direction*half*x[j];
        sum+=w[j]*std::exp(logLocalRate(sign*segment.value(q),sign*segment.derivative(q)));
    }
    if (!std::isfinite(sum)) throw std::runtime_error("nonfinite causal local hazard");
    return half*sum;
}
struct Fragment { double a,b,mass; };
std::vector<Fragment> fragments(const RayMeanProfile& p,int sign,double a,double b,double fine) {
    std::vector<double> knots{a,b};
    for (double i=std::floor(a/fine)+1;i*fine<b;++i) if (i*fine>a) knots.push_back(i*fine);
    const auto& segments=p.segments();
    auto it=std::upper_bound(segments.begin(),segments.end(),a,[](double q,const auto& s){return q<s.end;});
    for (;it!=segments.end() && it->end<b;++it) if (it->end>a) knots.push_back(it->end);
    std::sort(knots.begin(),knots.end());
    knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
    std::vector<Fragment> result;
    for (std::size_t i=1;i<knots.size();++i) result.push_back({knots[i-1],knots[i],integral(p,sign,knots[i-1],knots[i])});
    return result;
}
double total(const std::vector<Fragment>& f) {
    double sum=0; for (const auto& v:f) sum+=v.mass; return sum;
}
void appendEntry(std::vector<float>& v,const RayMeanProfile& p,int sign) {
    const auto& s=p.segments().front();
    v.push_back(static_cast<float>(std::asinh(sign*s.value(0))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(0))));
}
void appendLeft(std::vector<float>& v,const RayMeanProfile& p,int sign,double q) {
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
    v.push_back(static_cast<float>(std::log1p(q)));
}
void appendToken(std::vector<float>& v,const RayMeanProfile& p,int sign,double a,double b) {
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).value(q))));
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).derivative(q))));
    v.push_back(static_cast<float>(std::log1p(b-a)));
}
void appendLocal(std::vector<float>& v,const RayMeanProfile& p,int sign,double q,double left,double step) {
    v.push_back(static_cast<float>(std::clamp((q-left)/step,0.,1.)));
    v.push_back(static_cast<float>(std::log1p(q)));
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
}
RenewalDenseWeights dense(const Json& root,const std::string& name,int input,int output,
    const std::string& weightSuffix=".weight",const std::string& biasSuffix=".bias") {
    if (input<1 || input>4096 || output<1 || output>4096) throw std::invalid_argument("invalid causal network size");
    const auto& w=root.at(name+weightSuffix); const auto& b=root.at(name+biasSuffix);
    RenewalDenseWeights result{input,output,w.at("values").get<std::vector<float>>(),b.at("values").get<std::vector<float>>()};
    if (w.at("shape")!=Json::array({output,input}) || b.at("shape")!=Json::array({output}) ||
        result.weight.size()!=static_cast<std::size_t>(input)*output || result.bias.size()!=static_cast<std::size_t>(output))
        throw std::invalid_argument("causal tensor shape mismatch: "+name);
    for (const auto* values:{&result.weight,&result.bias}) for (float x:*values)
        if (!std::isfinite(x)) throw std::invalid_argument("nonfinite causal weight");
    return result;
}
Vec linear(const RenewalDenseWeights& w,const Vec& x) {
    return Eigen::Map<const Mat>(w.weight.data(),w.output,w.input)*x+Eigen::Map<const Vec>(w.bias.data(),w.output);
}
Vec silu(Vec x) { for (int i=0;i<x.size();++i) x[i]=x[i]/(1.f+std::exp(-x[i])); return x; }
float softplus(float x) { return x>20.f ? x : std::log1p(std::exp(x)); }
class ScalarBackend final : public CausalFirstHitBackend {
    const CausalNetworkWeights& w_;
    std::vector<Vec> states_;
    Vec input(int slot,const std::vector<float>& local,std::size_t row,int columns) const {
        Vec v(w_.hidden+columns); v.head(w_.hidden)=states_.at(slot);
        v.tail(columns)=Eigen::Map<const Vec>(local.data()+row*columns,columns); return v;
    }
public:
    explicit ScalarBackend(const CausalNetworkWeights& w):w_(w) {}
    void initialize(const std::vector<float>& entries) override {
        states_.clear(); states_.reserve(entries.size()/2);
        for (std::size_t i=0;i<entries.size();i+=2)
            states_.push_back(linear(w_.initial2,silu(linear(w_.initial0,Eigen::Map<const Vec>(entries.data()+i,2)))));
    }
    std::vector<float> correction(const std::vector<int>& slots,const std::vector<float>& left) override {
        std::vector<float> out;
        for (std::size_t i=0;i<slots.size();++i) out.push_back(linear(w_.head2,silu(linear(w_.head0,input(slots[i],left,i,3))))[0]);
        return out;
    }
    void advance(const std::vector<int>& slots,const std::vector<float>& tokens) override {
        for (std::size_t i=0;i<slots.size();++i) {
            const Vec embedded=silu(linear(w_.segment2,silu(linear(w_.segment0,Eigen::Map<const Vec>(tokens.data()+7*i,7)))));
            const Vec a=linear(w_.gruInput,embedded),b=linear(w_.gruHidden,states_.at(slots[i]));
            auto& h=states_[slots[i]];
            for (int j=0;j<w_.hidden;++j) {
                const float r=1.f/(1.f+std::exp(-a[j]-b[j]));
                const float z=1.f/(1.f+std::exp(-a[j+w_.hidden]-b[j+w_.hidden]));
                const float n=std::tanh(a[j+2*w_.hidden]+r*b[j+2*w_.hidden]);
                h[j]=(1-z)*n+z*h[j];
            }
            if (!h.allFinite()) throw std::runtime_error("nonfinite causal GRU state");
        }
    }
    std::vector<float> mixture(const std::vector<int>& slots,const std::vector<float>& local) override {
        std::vector<float> out;
        for (std::size_t i=0;i<slots.size();++i) {
            Vec raw=linear(w_.mixture4,silu(linear(w_.mixture2,silu(linear(w_.mixture0,input(slots[i],local,i,4))))));
            for (int j=2*w_.components;j<3*w_.components;++j) raw[j]=softplus(raw[j])+.01f;
            out.insert(out.end(),raw.data(),raw.data()+raw.size());
        }
        return out;
    }
};
RenewalSpeedMixture decode(const float* raw,int components,double signedDerivative) {
    RenewalSpeedMixture m;
    const double maximum=*std::max_element(raw,raw+components); double sum=0;
    for (int j=0;j<components;++j) {
        m.weights.push_back(std::exp(raw[j]-maximum)); sum+=m.weights.back();
        m.means.push_back(-signedDerivative+raw[components+j]); m.scales.push_back(raw[2*components+j]);
        if (!std::isfinite(m.means.back()) || !(m.scales.back()>0) || !std::isfinite(m.scales.back()))
            throw std::runtime_error("invalid causal speed parameters");
    }
    if (!(sum>0) || !std::isfinite(sum)) throw std::runtime_error("invalid causal mixture weights");
    for (auto& x:m.weights) x/=sum; return m;
}
double sampleSpeed(const RenewalSpeedMixture& m,double component,double quantile) {
    std::size_t j=0; double cumulative=m.weights[0];
    while (j+1<m.weights.size() && component>=cumulative) cumulative+=m.weights[++j];
    const double logTail=std::log1p(-quantile)+normalLogCdf(m.means[j]/m.scales[j]);
    return std::max(0.,m.means[j]-m.scales[j]*normalQuantileFromLogCdf(logTail));
}
}

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

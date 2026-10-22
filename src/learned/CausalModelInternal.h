#pragma once
// Shared fixed-grid integration and scalar network for first-zero and A/B models.
#include "macrofacet/learned/RenewalHazardModel.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "CausalFirstHitBackend.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>

namespace mf::causal_detail {
using Json=nlohmann::json;
using Vec=Eigen::VectorXf;
using Mat=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
inline double logAdd(double a,double b) {
    const double m=std::max(a,b);
    return std::isinf(m) ? m : m+std::log(std::exp(a-m)+std::exp(b-m));
}
inline double logLocalRate(double b,double d) {
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
inline const RayMeanSegment& segmentAt(const RayMeanProfile& p,double q) {
    const auto& s=p.segments();
    auto it=std::lower_bound(s.begin(),s.end(),q,[](const auto& a,double x){return a.end<x;});
    return it==s.end() ? s.back() : *it;
}
inline void validateProfile(const RayMeanProfile& p) {
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
inline double integral(const RayMeanProfile& p,int sign,double a,double b) {
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
inline std::vector<Fragment> fragments(const RayMeanProfile& p,int sign,double a,double b,double fine) {
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
inline double total(const std::vector<Fragment>& f) {
    double sum=0; for (const auto& v:f) sum+=v.mass; return sum;
}
inline void appendEntry(std::vector<float>& v,const RayMeanProfile& p,int sign) {
    const auto& s=p.segments().front();
    v.push_back(static_cast<float>(std::asinh(sign*s.value(0))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(0))));
}
inline void appendLeft(std::vector<float>& v,const RayMeanProfile& p,int sign,double q) {
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
    v.push_back(static_cast<float>(std::log1p(q)));
}
inline void appendToken(std::vector<float>& v,const RayMeanProfile& p,int sign,double a,double b) {
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).value(q))));
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).derivative(q))));
    v.push_back(static_cast<float>(std::log1p(b-a)));
}
inline void appendLocal(std::vector<float>& v,const RayMeanProfile& p,int sign,double q,double left,double step) {
    v.push_back(static_cast<float>(std::clamp((q-left)/step,0.,1.)));
    v.push_back(static_cast<float>(std::log1p(q)));
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
}
inline RenewalDenseWeights dense(const Json& root,const std::string& name,int input,int output,
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
inline Vec linear(const RenewalDenseWeights& w,const Vec& x) {
    return Eigen::Map<const Mat>(w.weight.data(),w.output,w.input)*x+Eigen::Map<const Vec>(w.bias.data(),w.output);
}
inline Vec silu(Vec x) { for (int i=0;i<x.size();++i) x[i]=x[i]/(1.f+std::exp(-x[i])); return x; }
inline float softplus(float x) { return x>20.f ? x : std::log1p(std::exp(x)); }
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
        const int columns=w_.initial0.input;
        if (entries.size()%columns) throw std::invalid_argument("invalid causal entry size");
        states_.clear(); states_.reserve(entries.size()/columns);
        for (std::size_t i=0;i<entries.size();i+=columns)
            states_.push_back(linear(w_.initial2,silu(linear(w_.initial0,Eigen::Map<const Vec>(entries.data()+i,columns)))));
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
inline RenewalSpeedMixture decode(const float* raw,int components,double signedDerivative) {
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
inline double sampleSpeed(const RenewalSpeedMixture& m,double component,double quantile) {
    std::size_t j=0; double cumulative=m.weights[0];
    while (j+1<m.weights.size() && component>=cumulative) cumulative+=m.weights[++j];
    const double logTail=std::log1p(-quantile)+normalLogCdf(m.means[j]/m.scales[j]);
    return std::max(0.,m.means[j]-m.scales[j]*normalQuantileFromLogCdf(logTail));
}
} // namespace mf::causal_detail

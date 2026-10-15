#include "macrofacet/learned/FirstHitSEModel.h"
#include "macrofacet/transport/RenewalRayDistribution.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

namespace mf {
namespace {
using Json=nlohmann::json;
using Vector=Eigen::VectorXf;
using Matrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
double softplus(double x) { return x>20 ? x : std::log1p(std::exp(x)); }
double logSoftplus(double x) { return x < -20 ? x : std::log(softplus(x)); }
double logAdd(double a,double b) {
    const double maximum=std::max(a,b);
    if (maximum==-std::numeric_limits<double>::infinity()) return maximum;
    return maximum+std::log(std::exp(a-maximum)+std::exp(b-maximum));
}
void silu(Vector& x,Eigen::ArrayXf& exponent) {
    // Use Eigen's packet exponential, as in RenewalHazardModel. The
    // nonpositive exponent keeps both sigmoid branches finite in the tails.
    exponent=(-x.array().abs()).exp();
    x.array()*=(x.array()>=0.f).select(1.f/(1.f+exponent),exponent/(1.f+exponent));
}
std::vector<float> tensor(const Json& weights,const std::string& key,std::vector<int> shape) {
    const auto& item=weights.at(key);
    const auto data=item.at("values").get<std::vector<float>>();
    std::size_t count=1;
    for (int n:shape) { if (n<1 || n>4096) throw std::invalid_argument("invalid first-hit dimension"); count*=n; }
    if (item.at("shape").get<std::vector<int>>()!=shape || data.size()!=count)
        throw std::invalid_argument("invalid first-hit tensor "+key);
    for (float x:data) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite first-hit tensor");
    return data;
}
struct Linear {
    Matrix weight; Vector bias;
    Vector operator()(const Vector& x) const { return weight*x+bias; }
    void evaluate(const Vector& x,Vector& output) const {
        output.resize(bias.size());
        output.noalias()=weight*x;
        output+=bias;
    }
};
Linear linear(const Json& weights,const std::string& name,int input,int output) {
    const auto w=tensor(weights,name+".weight",{output,input}), b=tensor(weights,name+".bias",{output});
    return {Eigen::Map<const Matrix>(w.data(),output,input),Eigen::Map<const Vector>(b.data(),output)};
}
struct Mlp {
    Linear a,b,c;
    void evaluate(const Vector& x,Vector& output) const {
        // Calls are nonrecursive. Scratch never escapes this call, and each
        // worker reuses its buffers across models without mutating weights.
        struct Scratch { Vector first,second; Eigen::ArrayXf exponent; };
        thread_local Scratch scratch;
        a.evaluate(x,scratch.first);
        silu(scratch.first,scratch.exponent);
        b.evaluate(scratch.first,scratch.second);
        silu(scratch.second,scratch.exponent);
        c.evaluate(scratch.second,output);
    }
    Vector operator()(const Vector& x) const {
        Vector output;
        evaluate(x,output);
        return output;
    }
};
Mlp mlp(const Json& weights,const std::string& name,int input,int width,int output) {
    return {linear(weights,name+".0",input,width),linear(weights,name+".2",width,width),linear(weights,name+".4",width,output)};
}
RayMeanSegment signedSegment(RayMeanSegment segment,int sign) {
    auto& p=segment.polynomial;
    p={sign*p.a,sign*p.b,sign*p.c,sign*p.d};
    return segment;
}
void validate(const RayMeanProfile& profile) {
    double value=profile.segments().front().value(0);
    for (const auto& segment:profile.segments()) {
        if (std::abs(segment.value(segment.begin)-value)>1e-8*(1+std::abs(value)))
            throw std::invalid_argument("first-zero mean must be continuous; use cubic profiles");
        value=segment.value(segment.end);
    }
}
struct State { Vector h; double H=0; };
struct Rate { Vector dh; double lambda; };
struct SplineState {
    Vector context;
    std::vector<double> x,y,derivatives;
    double logEscape=0,logHit=0;
};
struct SplineValue { double value,logDerivative,tail; };
SplineValue rationalQuadratic(const SplineState& s,double value,bool inverse=false) {
    value=std::clamp(value,0.,1.);
    const auto& knots=inverse ? s.y : s.x;
    const auto upper=std::upper_bound(knots.begin(),knots.end(),value);
    const auto index=std::min(static_cast<std::size_t>(upper-knots.begin()-1),knots.size()-2);
    const double x0=s.x[index],y0=s.y[index],w=s.x[index+1]-x0,h=s.y[index+1]-y0;
    const double d0=s.derivatives[index],d1=s.derivatives[index+1],slope=h/w,bend=d0+d1-2*slope;
    double theta;
    if (inverse) {
        const double offset=std::max(0.,value-y0),a=offset*bend+h*(slope-d0);
        const double b=h*d0-offset*bend,c=-slope*offset;
        theta=std::clamp(2*c/(-b-std::sqrt(std::max(0.,b*b-4*a*c))),0.,1.);
    } else theta=std::clamp((value-x0)/w,0.,1.);
    const double cross=theta*(1-theta),denominator=slope+bend*cross;
    const double logDerivative=2*std::log(slope)+std::log(d1*theta*theta+2*slope*cross+d0*(1-theta)*(1-theta))
        -2*std::log(denominator);
    if (inverse) return {x0+w*theta,-logDerivative,0};
    return {y0+h*(slope*theta*theta+d0*cross)/denominator,logDerivative,
        std::max(0.,1-s.y[index+1]+h*(slope*(1-theta)*(1-theta)+d1*cross)/denominator)};
}
}

struct FirstHitSEModel::Impl {
    std::string architecture;
    int hidden=0,components=0;
    double step=0,floor=0;
    Mlp initial,dynamics,hazard,mixture;
    Mlp fixedEncoder;
    Linear cdfHead;
    int profilePoints=0,bins=0;
    double minimumWidth=0,minimumHeight=0,minimumDerivative=0;
    std::optional<RenewalHazardModel> gru;
    SplineState encodeSpline(const RayMeanProfile& profile,int sign) const {
        Vector input(3*profilePoints+1);
        const auto& segments=profile.segments();
        std::size_t index=0;
        for (int j=0;j<profilePoints;++j) {
            const double u=static_cast<double>(j)/(profilePoints-1),q=std::min(profile.maximumX(),u*profile.maximumX());
            while (index+1<segments.size() && segments[index].end<q) ++index;
            input[3*j]=static_cast<float>(std::asinh(sign*segments[index].value(q)));
            input[3*j+1]=static_cast<float>(std::asinh(sign*segments[index].derivative(q)));
            input[3*j+2]=static_cast<float>(u);
        }
        input[3*profilePoints]=static_cast<float>(std::log1p(profile.maximumX()));
        SplineState state;
        state.context=fixedEncoder(input);
        const auto raw=cdfHead(state.context);
        if (!raw.allFinite()) throw std::runtime_error("nonfinite first-hit spline parameters");
        state.logEscape=-softplus(-raw[0]); state.logHit=-softplus(raw[0]);
        const auto partition=[&](int offset,double floor) {
            double sum=0;
            const double maximum=raw.segment(offset,bins).maxCoeff();
            std::vector<double> values;
            for (int j=0;j<bins;++j) { values.push_back(std::exp(raw[offset+j]-maximum)); sum+=values.back(); }
            std::vector<double> knots{0};
            for (int j=0;j<bins-1;++j) knots.push_back(knots.back()+floor+(1-bins*floor)*values[j]/sum);
            knots.push_back(1);
            return knots;
        };
        state.x=partition(1,minimumWidth); state.y=partition(1+bins,minimumHeight);
        for (int j=0;j<=bins;++j) state.derivatives.push_back(minimumDerivative+softplus(raw[1+2*bins+j]));
        return state;
    }
    RenewalSpeedMixture splineMixture(const SplineState& state,const RayMeanSegment& segment,
        double q,double horizon,int sign) const {
        Vector input(hidden+4);
        input << state.context,static_cast<float>(q/horizon),static_cast<float>(std::log1p(q)),
            static_cast<float>(std::asinh(sign*segment.value(q))),static_cast<float>(std::asinh(sign*segment.derivative(q)));
        return speedMixture(mixture(input),sign*segment.derivative(q));
    }
    State initialize(const RayMeanProfile& profile,int sign) const {
        const auto& first=profile.segments().front();
        Vector input(2);
        input << static_cast<float>(std::asinh(sign*first.value(0))),static_cast<float>(std::asinh(sign*first.derivative(0)));
        return {initial(input),0};
    }
    void input(const Vector& h,const RayMeanSegment& segment,double q,int sign,Vector& result) const {
        q=std::clamp(q,segment.begin,segment.end);
        result.resize(hidden+3);
        result << h,static_cast<float>(std::asinh(sign*segment.value(q))),
            static_cast<float>(std::asinh(sign*segment.derivative(q))),static_cast<float>(std::log1p(q));
    }
    Vector input(const Vector& h,const RayMeanSegment& segment,double q,int sign) const {
        Vector result;
        input(h,segment,q,sign,result);
        return result;
    }
    void rhs(const Vector& h,const RayMeanSegment& segment,double q,int sign,Rate& rate) const {
        struct Scratch { Vector input,hazard; };
        thread_local Scratch scratch;
        input(h,segment,q,sign,scratch.input);
        hazard.evaluate(scratch.input,scratch.hazard);
        const float raw=scratch.hazard[0];
        // Match float32 PyTorch activations before double cumulative quadrature.
        rate.lambda=static_cast<float>(softplus(raw));
        dynamics.evaluate(scratch.input,rate.dh);
    }
    State advance(const State& state,const RayMeanSegment& segment,double q,double dt,int sign) const {
        // All four slopes must remain distinct until the RK4 combination.
        // This nonrecursive workspace is local to each rendering thread.
        struct Scratch { Rate k1,k2,k3,k4; Vector temporary; };
        thread_local Scratch scratch;
        auto& k1=scratch.k1; auto& k2=scratch.k2;
        auto& k3=scratch.k3; auto& k4=scratch.k4;
        rhs(state.h,segment,q,sign,k1);
        scratch.temporary=state.h+static_cast<float>(dt/2)*k1.dh;
        rhs(scratch.temporary,segment,q+dt/2,sign,k2);
        scratch.temporary=state.h+static_cast<float>(dt/2)*k2.dh;
        rhs(scratch.temporary,segment,q+dt/2,sign,k3);
        scratch.temporary=state.h+static_cast<float>(dt)*k3.dh;
        rhs(scratch.temporary,segment,q+dt,sign,k4);
        State result{state.h+static_cast<float>(dt/6)*(k1.dh+2*k2.dh+2*k3.dh+k4.dh),
            state.H+(dt/6)*(k1.lambda+2*k2.lambda+2*k3.lambda+k4.lambda)};
        if (!result.h.allFinite() || !std::isfinite(result.H)) throw std::runtime_error("nonfinite first-hit ODE state");
        return result;
    }
    RenewalSpeedMixture distribution(const State& state,const RayMeanSegment& segment,double q,int sign) const {
        const Vector p=mixture(input(state.h,segment,q,sign));
        return speedMixture(p,sign*segment.derivative(q));
    }
    RenewalSpeedMixture speedMixture(const Vector& p,double signedDerivative) const {
        if (!p.allFinite()) throw std::runtime_error("nonfinite first-hit mixture");
        RenewalSpeedMixture result;
        const double maximum=p.head(components).maxCoeff();
        double total=0;
        for (int i=0;i<components;++i) {
            const double weight=std::exp(p[i]-maximum);
            total+=weight; result.weights.push_back(weight);
            result.means.push_back(-signedDerivative+p[components+i]);
            result.scales.push_back(static_cast<float>(softplus(p[2*components+i]))+floor);
        }
        for (auto& w:result.weights) w/=total;
        return result;
    }
};

FirstHitSEModel FirstHitSEModel::load(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open first-hit model "+path.string());
    Json root; stream >> root;
    if (root.at("format")!="macrofacet.first_hit_se" || root.at("version")!=1 ||
        root.at("event")!="unconditioned_first_zero" || root.at("profile_mode")!="cubic" ||
        root.at("kernel")!=Json{{"type","squared_exponential"},{"parameterization","unit_length"},{"beta",1.0}} ||
        root.at("branch_order")!=Json::array({1,-1}) ||
        root.at("activation_dtype")!="float32" ||
        root.at("distance_convention")!="q=distance_from_entry/ell" ||
        root.at("mean_convention")!="b=m/sigma" ||
        root.at("gradient_convention")!="signed_derivative=-initial_sign*sigma/ell*W")
        throw std::invalid_argument("unsupported first-hit model contract");
    const auto& c=root.at("model_config");
    auto impl=std::make_shared<Impl>();
    impl->architecture=c.at("architecture"); impl->hidden=c.at("hidden"); impl->components=c.at("components");
    if (root.at("solver")!=(impl->architecture=="spline" ? "analytic_rqs" : "rk4"))
        throw std::invalid_argument("first-hit solver does not match its architecture");
    impl->step=c.at("maximum_step"); impl->floor=c.at("sigma_floor");
    if (!(impl->step>0 && impl->step<=1 && impl->floor>0 && impl->floor<1))
        throw std::invalid_argument("invalid first-hit step or scale floor");
    if (impl->hidden<1 || impl->hidden>4096 || impl->components<1 || impl->components>1024)
        throw std::invalid_argument("invalid first-hit network dimensions");
    if (impl->architecture=="spline") {
        if (root.at("fixed_profile_convention")!="uniform_q_over_Q_asinh_signed_b_db_u_log1p_Q")
            throw std::invalid_argument("unsupported fixed first-hit profile convention");
        impl->profilePoints=c.at("profile_points"); impl->bins=c.at("bins");
        impl->minimumWidth=c.at("minimum_bin_width"); impl->minimumHeight=c.at("minimum_bin_height");
        impl->minimumDerivative=c.at("minimum_derivative");
        if (impl->profilePoints<2 || impl->profilePoints>1024 || impl->bins<2 || impl->bins>256 ||
            !(impl->minimumWidth>0 && impl->bins*impl->minimumWidth<1 &&
              impl->minimumHeight>0 && impl->bins*impl->minimumHeight<1 &&
              impl->minimumDerivative>0 && impl->minimumDerivative<1))
            throw std::invalid_argument("invalid fixed-grid spline configuration");
        const auto& weights=root.at("weights");
        const int width=c.at("width"),mixtureWidth=c.at("mixture_width");
        impl->fixedEncoder=mlp(weights,"encoder",3*impl->profilePoints+1,width,impl->hidden);
        impl->cdfHead=linear(weights,"cdf_head",impl->hidden,3*impl->bins+2);
        impl->mixture=mlp(weights,"mixture_head",impl->hidden+4,mixtureWidth,3*impl->components);
    } else if (impl->architecture=="gru") {
        std::istringstream nested(root.at("gru").dump());
        impl->gru=RenewalHazardModel::load(nested);
        if (!impl->gru->hasMixture() || impl->gru->kernelType()!=CovarianceKernelType::SquaredExponential)
            throw std::invalid_argument("first-hit GRU requires a complete SE model");
        const auto& nestedConfig=root.at("gru").at("model_config");
        if (nestedConfig.at("hidden")!=c.at("hidden") || nestedConfig.at("components")!=c.at("components") ||
            nestedConfig.at("embedding")!=c.at("width") || nestedConfig.at("hazard_width")!=c.at("width") ||
            nestedConfig.at("mixture_width")!=c.at("mixture_width") || nestedConfig.at("sigma_floor")!=c.at("sigma_floor"))
            throw std::invalid_argument("first-hit GRU dimensions disagree with its wrapper");
    } else if (impl->architecture=="rno") {
        const auto& weights=root.at("weights");
        const int width=c.at("width"), mixtureWidth=c.at("mixture_width"), h=impl->hidden;
        impl->initial=mlp(weights,"initial_encoder",2,width,h);
        impl->dynamics=mlp(weights,"dynamics",h+3,width,h);
        impl->hazard=mlp(weights,"hazard_head",h+3,width,1);
        impl->mixture=mlp(weights,"mixture_head",h+3,mixtureWidth,3*impl->components);
    } else throw std::invalid_argument("unknown first-hit architecture");
    return FirstHitSEModel(std::move(impl));
}

const std::string& FirstHitSEModel::architecture() const { return impl_->architecture; }
double FirstHitSEModel::maximumStep() const { return impl_->step; }

std::vector<FirstHitSEPrediction> FirstHitSEModel::query(const RayMeanProfile& profile,
    const std::vector<double>& queries,double maximumStep) const {
    validate(profile);
    const double step=maximumStep==0 ? impl_->step : maximumStep;
    if (!(step>0) || !std::isfinite(step)) throw std::invalid_argument("invalid first-hit query step");
    std::vector<std::size_t> order(queries.size());
    std::iota(order.begin(),order.end(),0);
    for (double q:queries) if (!(q>=0 && q<=profile.maximumX()) || !std::isfinite(q))
        throw std::invalid_argument("first-hit query outside profile");
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b) { return queries[a]<queries[b]; });
    std::vector<FirstHitSEPrediction> result(queries.size());
    const auto& segments=profile.segments();
    for (int branch=0;branch<2;++branch) {
        const int sign=branch==0 ? 1 : -1;
        std::size_t index=0;
        double q=0,prefix=0;
        State state;
        std::optional<SplineState> spline;
        RenewalHazardModel::State gruState;
        RenewalHazardModel::Step gruStep;
        auto segment=signedSegment(segments[0],sign);
        if (impl_->architecture=="spline") spline=impl_->encodeSpline(profile,sign);
        else if (impl_->gru) {
            gruState=impl_->gru->initialize(segment,{});
            gruStep=impl_->gru->evaluate(gruState,segment);
        } else state=impl_->initialize(profile,sign);
        for (auto i:order) {
            const double target=queries[i];
            if (spline) {
                const auto value=rationalQuadratic(*spline,target/profile.maximumX());
                const double logT=logAdd(spline->logEscape,spline->logHit+std::log(std::max(1e-300,value.tail)));
                result[i].cumulative[branch]=-logT;
                result[i].logHazard[branch]=spline->logHit+value.logDerivative-std::log(profile.maximumX())-logT;
                while (index+1<segments.size() && segments[index].end<target) ++index;
                result[i].mixture[branch]=impl_->splineMixture(*spline,segments[index],target,profile.maximumX(),sign);
            } else if (impl_->gru) {
                while (target>segment.end && index+1<segments.size()) {
                    prefix+=RenewalHazardSegment(segment.begin,segment.end,gruStep.rates).cumulative(1);
                    gruState=std::move(gruStep.nextState);
                    segment=signedSegment(segments[++index],sign);
                    gruStep=impl_->gru->evaluate(gruState,segment);
                }
                const double u=std::clamp((target-segment.begin)/(segment.end-segment.begin),0.,1.);
                const RenewalHazardSegment hazard(segment.begin,segment.end,gruStep.rates);
                result[i].cumulative[branch]=prefix+hazard.cumulative(u);
                result[i].logHazard[branch]=std::log(hazard.hazard(u));
                result[i].mixture[branch]=impl_->gru->mixture(gruState,segment,u);
            } else {
                // Query locations never split persistent integration steps.
                // The same nodes are used by collision sampling.
                while (target>std::min(q+step,segments[index].end)) {
                    const double right=std::min(q+step,segments[index].end);
                    state=impl_->advance(state,segments[index],q,right-q,sign);
                    q=right;
                    if (q==segments[index].end && index+1<segments.size()) ++index;
                }
                const auto atQuery=impl_->advance(state,segments[index],q,target-q,sign);
                result[i].cumulative[branch]=atQuery.H;
                result[i].logHazard[branch]=logSoftplus(impl_->hazard(impl_->input(atQuery.h,segments[index],target,sign))[0]);
                result[i].mixture[branch]=impl_->distribution(atQuery,segments[index],target,sign);
            }
        }
    }
    const double b0=segments[0].value(0);
    const double lp=normalLogCdf(b0), ln=normalLogCdf(-b0), normalization=logAdd(lp,ln);
    for (std::size_t i=0;i<result.size();++i) {
        auto& r=result[i]; r.q=queries[i];
        const double a=lp-normalization-r.cumulative[0], b=ln-normalization-r.cumulative[1];
        r.logTransmittance=logAdd(a,b);
        r.logDensity=logAdd(a+r.logHazard[0],b+r.logHazard[1]);
        r.hitBranchProbability={std::exp(a+r.logHazard[0]-r.logDensity),std::exp(b+r.logHazard[1]-r.logDensity)};
    }
    return result;
}

FirstHitSESample FirstHitSEModel::sample(const RayMeanProfile& profile,Random& rng,
    double maximumStep,double tolerance) const {
    validate(profile);
    const double step=maximumStep==0 ? impl_->step : maximumStep;
    if (!(step>0 && tolerance>0) || !std::isfinite(step) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid first-hit sampling tolerances");
    const auto& segments=profile.segments();
    const int sign=std::log(rng.openUniform01())<normalLogCdf(segments[0].value(0)) ? 1 : -1;
    if (impl_->architecture=="spline") {
        const auto spline=impl_->encodeSpline(profile,sign);
        const double uniform=rng.openUniform01(),escape=std::exp(spline.logEscape);
        if (uniform<=escape) return {false,profile.maximumX(),0,sign};
        const double u=std::clamp((uniform-escape)/std::exp(spline.logHit),0.,1.);
        const double q=profile.maximumX()*rationalQuadratic(spline,u,true).value;
        auto it=std::lower_bound(segments.begin(),segments.end(),q,[](const RayMeanSegment& s,double x) { return s.end<x; });
        if (it==segments.end() || !(q>=0) || !std::isfinite(q)) throw std::runtime_error("invalid spline first-hit sample");
        return {true,q,impl_->splineMixture(spline,*it,q,profile.maximumX(),sign).sample(rng),sign};
    }
    double target=-std::log(rng.openUniform01());
    if (impl_->gru) {
        auto state=impl_->gru->initialize(signedSegment(segments[0],sign),{});
        for (const auto& original:segments) {
            const auto segment=signedSegment(original,sign);
            auto evaluated=impl_->gru->evaluate(state,segment);
            const RenewalHazardSegment hazard(segment.begin,segment.end,evaluated.rates);
            const double mass=hazard.cumulative(1);
            if (mass>0 && target<=mass) {
                const double u=hazard.inverse(target,0,1,tolerance);
                return {true,segment.begin+(segment.end-segment.begin)*u,impl_->gru->mixture(state,segment,u).sample(rng),sign};
            }
            target-=mass; state=std::move(evaluated.nextState);
        }
    } else {
        auto state=impl_->initialize(profile,sign);
        for (const auto& segment:segments) {
            double q=segment.begin;
            while (q<segment.end) {
                const double dt=std::min(step,segment.end-q);
                auto next=impl_->advance(state,segment,q,dt,sign);
                if (next.H>=target && next.H>state.H) {
                    double lo=0,hi=dt;
                    for (int iteration=0;iteration<100 && hi-lo>tolerance;++iteration) {
                        const double mid=(lo+hi)/2;
                        if (impl_->advance(state,segment,q,mid,sign).H<target) lo=mid; else hi=mid;
                    }
                    const double local=(lo+hi)/2, hit=q+local;
                    const auto atHit=impl_->advance(state,segment,q,local,sign);
                    return {true,hit,impl_->distribution(atHit,segment,hit,sign).sample(rng),sign};
                }
                state=std::move(next); q=dt==segment.end-q ? segment.end : q+dt;
            }
        }
    }
    return {false,profile.maximumX(),0,sign};
}
} // namespace mf

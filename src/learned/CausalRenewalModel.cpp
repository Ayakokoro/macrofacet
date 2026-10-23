#include "macrofacet/learned/CausalRenewalModel.h"
#include "CausalModelInternal.h"
#include <fstream>

namespace mf {
using namespace causal_detail;
namespace {
std::vector<float> renewalEntry(const RayMeanProfile& p, const RayStartCondition& start) {
    if (start.mode != RayStartMode::PositiveExterior && start.mode != RayStartMode::SurfaceOutward)
        throw std::invalid_argument("invalid causal renewal start mode");
    const bool surface = start.mode == RayStartMode::SurfaceOutward;
    if (surface && (!(start.outwardDerivative > 0) || !std::isfinite(start.outwardDerivative)))
        throw std::invalid_argument("causal renewal B requires a finite positive outward derivative");
    const auto& first = p.segments().front();
    const double b0 = first.value(0);
    std::vector<float> entry;
    for (double x : {surface ? 1. : 0., b0, surface ? -b0 : 0.,
                     surface ? start.outwardDerivative-first.derivative(0) : 0.}) {
        entry.push_back(static_cast<float>(std::asinh(x)));
        if (!std::isfinite(entry.back())) throw std::invalid_argument("nonfinite causal renewal entry");
    }
    return entry;
}
double correctionFactor(double delta) {
    const double factor = std::exp(delta);
    if (!(factor > 0) || !std::isfinite(factor))
        throw std::runtime_error("nonfinite causal renewal hazard correction");
    return factor;
}
}

struct CausalRenewalModel::Impl {
    CausalNetworkWeights weights;
    double step = 0, fine = 0;
    std::string checkpoint;
};

CausalRenewalModel CausalRenewalModel::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open causal renewal model " + path.string());
    return load(input);
}

CausalRenewalModel CausalRenewalModel::load(std::istream& input) {
    Json r; input >> r;
    const Json conditioning = {
        {"modes", {"positive_exterior", "collision_state"}},
        {"initial", {"surface_mode", "b0", "known_value_residual", "known_derivative_residual"}},
        {"derivative_residual", "ell*g0/sigma - b_prime(0)"},
        {"encoding", "asinh; unknown A residuals masked"},
        {"event", "first_downcrossing_after_start"}, {"baseline", "positive_side_local_hazard"}};
    const Json target = {
        {"variable", "w = -ell * directional_derivative_at_hit / sigma"},
        {"support", "w > 0; no upper truncation"}, {"physical_derivative", "-sigma * w / ell"},
        {"event", "first_downcrossing_after_start"}, {"state", "left_limit_of_fixed_q_interval"}};
    if (r.at("format") != "macrofacet.causal_renewal" || r.at("version") != 1 ||
        r.at("kernel") != "squared_exponential" || r.at("profile_mode") != "cubic" ||
        r.at("activation_dtype") != "float32" || r.at("conditioning") != conditioning || r.at("target") != target ||
        r.at("gru_convention") != "pytorch_rzn_reset_after" ||
        r.at("distance_convention") != "q=distance_from_birth/ell" || r.at("mean_convention") != "b=m/sigma" ||
        r.at("state_convention") != "entering_fixed_q_interval_left_limit_gradient")
        throw std::invalid_argument("unsupported causal renewal model contract");
    auto p = std::make_shared<Impl>();
    const auto& c = r.at("model_config"); const auto& g = r.at("gradient_config");
    auto& w = p->weights;
    w.hidden = c.at("hidden"); w.embedding = c.at("embedding"); w.components = g.at("size");
    const int width = c.at("width"), gw = g.at("width"), h = w.hidden, e = w.embedding;
    p->step = c.at("step"); p->fine = c.at("integration_step");
    if (!(p->step > 0 && p->step <= 1 && p->fine > 0 && p->fine <= p->step) || c.at("quadrature_order") != 8 ||
        c.at("history") != true || g.at("history") != true || g.at("kind") != "mixture" || g.at("hidden") != h ||
        h < 1 || h > 1024 || e < 1 || e > 1024 || w.components < 1 || w.components > 1024)
        throw std::invalid_argument("invalid causal renewal network configuration");
    const auto& b = r.at("backbone"); const auto& m = r.at("gradient");
    w.initial0 = dense(b,"initial.0",4,h); w.initial2 = dense(b,"initial.2",h,h);
    w.segment0 = dense(b,"segment.0",7,e); w.segment2 = dense(b,"segment.2",e,e);
    w.head0 = dense(b,"head.0",h+3,width); w.head2 = dense(b,"head.2",width,1);
    w.gruInput = dense(b,"gru",e,3*h,".weight_ih_l0",".bias_ih_l0");
    w.gruHidden = dense(b,"gru",h,3*h,".weight_hh_l0",".bias_hh_l0");
    w.mixture0 = dense(m,"net.0",h+4,gw); w.mixture2 = dense(m,"net.2",gw,gw);
    w.mixture4 = dense(m,"net.4",gw,3*w.components);
    p->checkpoint = r.at("checkpoint_sha256").get<std::string>();
    return CausalRenewalModel(std::move(p));
}

double CausalRenewalModel::step() const { return impl_->step; }
double CausalRenewalModel::integrationStep() const { return impl_->fine; }
const std::string& CausalRenewalModel::checkpointSha256() const { return impl_->checkpoint; }
const CausalNetworkWeights& CausalRenewalModel::networkWeights() const { return impl_->weights; }

std::vector<CausalRenewalPrediction> CausalRenewalModel::query(const RayMeanProfile& p,
    const RayStartCondition& start, const std::vector<double>& q) const {
    validateProfile(p);
    ScalarBackend backend(impl_->weights); backend.initialize(renewalEntry(p,start));
    for (double x : q) if (!(x >= 0 && x <= p.maximumX()) || !std::isfinite(x))
        throw std::invalid_argument("causal renewal query outside profile");
    std::vector<CausalRenewalPrediction> result(q.size());
    if (q.empty()) return result;
    const auto& m = *impl_;
    const auto count = static_cast<std::size_t>(std::ceil(p.maximumX()/m.step));
    std::vector<double> prefix{0}, deltas;
    for (std::size_t cell = 0; cell < count; ++cell) {
        const double a = cell*m.step, b = std::min((cell+1)*m.step,p.maximumX());
        std::vector<float> left; appendLeft(left,p,1,a);
        const double delta = backend.correction({0},left)[0]; deltas.push_back(delta);
        prefix.push_back(prefix.back()+correctionFactor(delta)*total(fragments(p,1,a,b,m.fine)));
        if (!std::isfinite(prefix.back())) throw std::runtime_error("nonfinite causal renewal cumulative hazard");
        std::vector<int> slots; std::vector<std::size_t> indices; std::vector<float> local;
        for (std::size_t j = 0; j < q.size(); ++j) {
            // A boundary's gradient uses the state before consuming that interval.
            const auto gradientCell = std::min(count-1,static_cast<std::size_t>(std::max(0.,std::ceil(q[j]/m.step)-1)));
            if (gradientCell == cell) { slots.push_back(0); indices.push_back(j); appendLocal(local,p,1,q[j],a,m.step); }
        }
        if (!slots.empty()) {
            const auto raw = backend.mixture(slots,local);
            for (std::size_t i = 0; i < indices.size(); ++i) {
                const auto j = indices[i];
                result[j].mixture = decode(raw.data()+i*3*m.weights.components,m.weights.components,
                                          segmentAt(p,q[j]).derivative(q[j]));
            }
        }
        if (cell+1 < count) { std::vector<float> token; appendToken(token,p,1,a,b); backend.advance({0},token); }
    }
    for (std::size_t j = 0; j < q.size(); ++j) {
        const auto cell = std::min(count-1,static_cast<std::size_t>(std::floor(q[j]/m.step)));
        auto& r = result[j]; r.q = q[j];
        r.cumulativeHazard = prefix[cell]+correctionFactor(deltas[cell])*total(fragments(p,1,cell*m.step,q[j],m.fine));
        const auto& s = segmentAt(p,q[j]); r.logHazard = logLocalRate(s.value(q[j]),s.derivative(q[j]))+deltas[cell];
    }
    return result;
}

CausalRenewalSample CausalRenewalModel::sample(const RayMeanProfile& p, const RayStartCondition& start,
    const std::array<double,3>& uniforms, double tolerance) const {
    validateProfile(p);
    if (!(tolerance > 0) || !std::isfinite(tolerance)) throw std::invalid_argument("invalid causal renewal tolerance");
    for (double u : uniforms) if (!(u > 0 && u < 1)) throw std::invalid_argument("causal renewal uniforms must be open");
    const auto& m = *impl_;
    ScalarBackend backend(m.weights); backend.initialize(renewalEntry(p,start));
    double remaining = -std::log(uniforms[0]);
    CausalRenewalSample result; result.q = p.maximumX();
    for (std::size_t cell = 0; cell*m.step < p.maximumX(); ++cell) {
        ++result.intervals;
        const double a = cell*m.step, b = std::min((cell+1)*m.step,p.maximumX());
        std::vector<float> left; appendLeft(left,p,1,a);
        const double factor = correctionFactor(backend.correction({0},left)[0]);
        const auto parts = fragments(p,1,a,b,m.fine);
        const double mass = factor*total(parts);
        if (!std::isfinite(mass)) throw std::runtime_error("nonfinite causal renewal integrated hazard");
        if (mass > 0 && remaining <= mass) {
            double residual = remaining/factor;
            const Fragment* selected = &parts.back();
            for (const auto& part : parts) {
                if ((residual <= part.mass && part.mass > 0) || &part == &parts.back()) { selected = &part; break; }
                residual -= part.mass;
            }
            residual = std::clamp(residual,0.,selected->mass);
            double lo = selected->a, hi = selected->b;
            for (int iteration = 0; iteration < 80 && hi-lo > tolerance; ++iteration) {
                const double mid = (lo+hi)/2;
                if (integral(p,1,selected->a,mid) < residual) lo = mid; else hi = mid;
            }
            result.hit = true; result.q = (lo+hi)/2;
            std::vector<float> local; appendLocal(local,p,1,result.q,a,m.step);
            const auto raw = backend.mixture({0},local);
            result.speed = sampleSpeed(decode(raw.data(),m.weights.components,segmentAt(p,result.q).derivative(result.q)),
                                       uniforms[1],uniforms[2]);
            if (!(result.speed > 0) || !std::isfinite(result.speed)) throw std::runtime_error("invalid causal renewal speed");
            return result;
        }
        remaining -= mass;
        if (b < p.maximumX()) { std::vector<float> token; appendToken(token,p,1,a,b); backend.advance({0},token); }
    }
    return result;
}
} // namespace mf

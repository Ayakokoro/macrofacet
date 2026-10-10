#include "macrofacet/learned/RenewalHazardModel.h"
#include "RenewalBatchBackend.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mf {
namespace {
using Json = nlohmann::json;
using Vector = Eigen::VectorXf;
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::ArrayXf sigmoid(const Eigen::ArrayXf& x) {
    // Packet exp avoids hundreds of scalar transcendental calls per segment.
    // The nonpositive exponent keeps both branches finite for large inputs.
    const Eigen::ArrayXf e = (-x.abs()).exp();
    return (x >= 0.0f).select(1.0f / (1.0f + e), e / (1.0f + e));
}
Vector silu(Vector x) {
    x.array() *= sigmoid(x.array());
    return x;
}
int dimension(const Json& config, const char* key) {
    const int value = config.at(key).get<int>();
    if (value < 1 || value > 4096) throw std::invalid_argument("invalid Renewal model dimension");
    return value;
}
std::vector<float> tensor(const Json& weights, const std::string& key, std::vector<int> shape) {
    const auto& entry = weights.at(key);
    if (entry.at("shape").get<std::vector<int>>() != shape)
        throw std::invalid_argument("Renewal tensor shape mismatch: " + key);
    std::size_t count = 1;
    for (int size : shape) count *= static_cast<std::size_t>(size);
    const auto values = entry.at("values").get<std::vector<float>>();
    if (values.size() != count) throw std::invalid_argument("Renewal tensor length mismatch: " + key);
    for (float x : values) if (!std::isfinite(x))
        throw std::invalid_argument("nonfinite Renewal tensor: " + key);
    return values;
}
Matrix matrix(const Json& weights, const std::string& key, int rows, int cols) {
    const auto values = tensor(weights, key, {rows, cols});
    return Eigen::Map<const Matrix>(values.data(), rows, cols);
}
Vector vector(const Json& weights, const std::string& key, int size) {
    const auto values = tensor(weights, key, {size});
    return Eigen::Map<const Vector>(values.data(), size);
}
struct Linear {
    Matrix weight;
    Vector bias;
    Vector operator()(const Vector& x) const { return weight*x + bias; }
};
Linear linear(const Json& weights, const std::string& name, int input, int output) {
    return {matrix(weights, name+".weight", output, input), vector(weights, name+".bias", output)};
}
struct Mlp {
    Linear first, second, last;
    Vector operator()(const Vector& x) const { return last(silu(second(silu(first(x))))); }
};
Mlp mlp(const Json& weights, const std::string& name, int input, int width, int output) {
    return {linear(weights, name+".0", input, width), linear(weights, name+".2", width, width),
            linear(weights, name+".4", width, output)};
}
} // namespace

struct RenewalHazardModel::Impl {
    CovarianceKernelType kernelType = CovarianceKernelType::Matern32;
    int hidden = 0, embedding = 0;
    int components = 0;
    float sigmaFloor = 0.01f;
    std::string checkpoint;
    Linear encoder0, encoder2;
    Mlp initial, hazard, mixture;
    Matrix weightInput, weightHidden;
    Vector biasInput, biasHidden;
};

RenewalHazardModel::RenewalHazardModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

RenewalHazardModel RenewalHazardModel::load(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open Renewal hazard bundle: " + path.string());
    Json root;
    stream >> root;
    const bool full = root.at("format") == "macrofacet.renewal";
    const auto& kernel = root.at("kernel");
    const bool matern32 = kernel == Json{{"type", "matern_3_2"}, {"parameterization", "unit_decay"}, {"beta", 1.0}};
    const bool se = kernel == Json{{"type", "squared_exponential"}, {"parameterization", "unit_length"}, {"beta", 1.0}};
    if ((!full && root.at("format") != "macrofacet.renewal_hazard") || root.at("version") != 1 ||
        (!matern32 && !se) ||
        root.at("activation_dtype") != "float32" || root.at("gru_convention") != "pytorch_rzn_reset_after" ||
        root.at("feature_transform") != "asinh_first_four_log_dx_identity" ||
        root.at("initial_transform") != "asinh_mode_b0_known_z0_known_d0")
        throw std::invalid_argument("unsupported Renewal model format, kernel or input convention");
    const auto& config = root.at("model_config");
    const auto& weights = root.at("weights");
    auto impl = std::make_shared<Impl>();
    impl->kernelType = se ? CovarianceKernelType::SquaredExponential : CovarianceKernelType::Matern32;
    impl->hidden = dimension(config, "hidden");
    impl->embedding = dimension(config, "embedding");
    const int width = dimension(config, "hazard_width");
    impl->checkpoint = root.at("checkpoint_sha256").get<std::string>();
    impl->encoder0 = linear(weights, "segment_encoder.0", 5, impl->embedding);
    impl->encoder2 = linear(weights, "segment_encoder.2", impl->embedding, impl->embedding);
    impl->initial = mlp(weights, "initial_encoder", 4, impl->hidden, impl->hidden);
    impl->hazard = mlp(weights, "hazard_head", impl->hidden+impl->embedding, width, 4);
    if (full) {
        if (root.at("mixture_convention") != "positive_truncated_components_residual_mean")
            throw std::invalid_argument("unsupported Renewal speed mixture convention");
        impl->components = dimension(config, "components");
        impl->sigmaFloor = config.at("sigma_floor").get<float>();
        if (!(impl->sigmaFloor > 0 && impl->sigmaFloor < 1))
            throw std::invalid_argument("invalid Renewal mixture scale floor");
        impl->mixture = mlp(weights, "mixture_head", impl->hidden+impl->embedding+3,
            dimension(config, "mixture_width"), 3*impl->components);
    }
    impl->weightInput = matrix(weights, "gru.weight_ih_l0", 3*impl->hidden, impl->embedding);
    impl->weightHidden = matrix(weights, "gru.weight_hh_l0", 3*impl->hidden, impl->hidden);
    impl->biasInput = vector(weights, "gru.bias_ih_l0", 3*impl->hidden);
    impl->biasHidden = vector(weights, "gru.bias_hh_l0", 3*impl->hidden);
    return RenewalHazardModel(std::move(impl));
}

RenewalHazardModel::State RenewalHazardModel::initialize(
    const RayMeanSegment& first, const RayStartCondition& start) const {
    const double b0 = first.value(first.begin);
    std::array<double, 4> known{0.0, b0, 0.0, 0.0};
    if (start.mode == RayStartMode::SurfaceOutward) {
        if (!(start.outwardDerivative > 0) || !std::isfinite(start.outwardDerivative))
            throw std::invalid_argument("Renewal surface start requires a positive known total derivative");
        known = {1.0, b0, -b0, start.outwardDerivative-first.derivative(first.begin)};
    } else if (start.mode != RayStartMode::PositiveExterior) {
        throw std::invalid_argument("invalid Renewal start mode");
    }
    Vector input(4);
    for (int i = 0; i < 4; ++i) input[i] = static_cast<float>(std::asinh(known[i]));
    State result = impl_->initial(input);
    if (!result.allFinite()) throw std::runtime_error("nonfinite Renewal initial state");
    return result;
}

RenewalHazardModel::Step RenewalHazardModel::evaluate(
    const State& state, const RayMeanSegment& segment) const {
    if (state.size() != impl_->hidden || !state.allFinite() || !(segment.end > segment.begin))
        throw std::invalid_argument("invalid Renewal state or segment");
    const Vector embedded = encodeSegment(segment);
    Vector context(impl_->hidden+impl_->embedding);
    context << state, embedded;
    const Vector logits = impl_->hazard(context);
    const Vector x = impl_->weightInput*embedded + impl_->biasInput;
    const Vector h = impl_->weightHidden*state + impl_->biasHidden;
    Step result;
    result.nextState.resize(impl_->hidden);
    const int n = impl_->hidden;
    const Eigen::ArrayXf gates = sigmoid(x.head(2*n).array()+h.head(2*n).array());
    // PyTorch resets the recurrent candidate AFTER its affine transform,
    // including bias_hh. Applying reset to the old state is a different GRU.
    result.nextState.array() = (1.0f-gates.tail(n)) *
        (x.tail(n).array()+gates.head(n)*h.tail(n).array()).tanh() + gates.tail(n)*state.array();
    for (int i = 0; i < 4; ++i) {
        const float v = logits[i];
        result.rates[i] = v > 20.0f ? v : std::log1p(std::exp(v));
        if (!std::isfinite(result.rates[i])) throw std::runtime_error("nonfinite Renewal hazard coefficient");
    }
    if (!result.nextState.allFinite()) throw std::runtime_error("nonfinite Renewal propagated state");
    return result;
}

RenewalHazardModel::State RenewalHazardModel::encodeSegment(const RayMeanSegment& segment) const {
    const auto features = segment.features();
    Vector input(5);
    for (int i = 0; i < 5; ++i)
        input[i] = static_cast<float>(i == 4 ? features[i] : std::asinh(features[i]));
    if (!input.allFinite()) throw std::invalid_argument("nonfinite Renewal segment features");
    return silu(impl_->encoder2(silu(impl_->encoder0(input))));
}

bool RenewalHazardModel::hasMixture() const { return impl_->components > 0; }

RenewalSpeedMixture RenewalHazardModel::mixture(const State& state,
    const RayMeanSegment& segment, double u) const {
    if (!hasMixture()) throw std::invalid_argument("normal sampling requires a full Renewal model export");
    if (state.size() != impl_->hidden || !state.allFinite() || !(u >= 0 && u <= 1))
        throw std::invalid_argument("invalid Renewal mixture state or coordinate");
    const double b = segment.polynomial.value(u);
    const double db = segment.polynomial.derivative(u)/(segment.end-segment.begin);
    Vector query(impl_->hidden+impl_->embedding+3);
    query << state, encodeSegment(segment), static_cast<float>(u),
        static_cast<float>(std::asinh(b)), static_cast<float>(std::asinh(db));
    const Vector parameters = impl_->mixture(query);
    if (!parameters.allFinite() || !std::isfinite(db))
        throw std::runtime_error("nonfinite Renewal mixture parameters");
    const int n = impl_->components;
    const float maximum = parameters.head(n).maxCoeff();
    RenewalSpeedMixture result;
    double total = 0;
    for (int i = 0; i < n; ++i) {
        const double weight = std::exp(static_cast<double>(parameters[i]-maximum));
        result.weights.push_back(weight); total += weight;
        result.means.push_back(-db+parameters[n+i]);
        const float raw = parameters[2*n+i];
        result.scales.push_back((raw > 20.f ? raw : std::log1p(std::exp(raw)))+impl_->sigmaFloor);
    }
    for (auto& weight : result.weights) weight /= total;
    return result;
}

double samplePositiveNormal(double mean, double scale, Random& rng) {
    if (!std::isfinite(mean) || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("invalid positive Gaussian parameters");
    const double a = -mean/scale;
    if (!std::isfinite(a)) throw std::invalid_argument("positive Gaussian standardized mean overflow");
    if (a <= 0) {
        for (;;) {
            const double z = rng.standardNormal();
            if (z > a) return scale*(z-a);
        }
    }
    // Exponential rejection in the tail. Sample the excess over the truncation
    // directly, avoiding subtraction of nearly equal large Gaussian values.
    const double rate = 0.5*a+0.5*std::hypot(a, 2.0);
    for (;;) {
        const double excess = -std::log(rng.openUniform01())/rate;
        const double residual = excess+(a-rate);
        if (std::log(rng.openUniform01()) <= -0.5*residual*residual)
            return scale*excess;
    }
}

double RenewalSpeedMixture::sample(Random& rng) const {
    if (weights.empty() || means.size() != weights.size() || scales.size() != weights.size())
        throw std::invalid_argument("invalid Renewal speed mixture dimensions");
    double total = 0;
    for (double weight : weights) {
        if (!(weight >= 0) || !std::isfinite(weight)) throw std::invalid_argument("invalid mixture weight");
        total += weight;
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::invalid_argument("invalid mixture mass");
    double target = rng.openUniform01()*total;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        target -= weights[i];
        if (target <= 0 || i+1 == weights.size()) return samplePositiveNormal(means[i], scales[i], rng);
    }
    throw std::runtime_error("unreachable mixture sample");
}

int RenewalHazardModel::hiddenSize() const { return impl_->hidden; }
CovarianceKernelType RenewalHazardModel::kernelType() const { return impl_->kernelType; }
const std::string& RenewalHazardModel::checkpointSha256() const { return impl_->checkpoint; }

std::shared_ptr<const RenewalNetworkWeights> RenewalHazardModel::batchWeights() const {
    if (!hasMixture()) throw std::invalid_argument("batched rendering requires a full Renewal model");
    auto result = std::make_shared<RenewalNetworkWeights>();
    const auto copy = [](const Matrix& w, const Vector& b) {
        return RenewalDenseWeights{static_cast<int>(w.cols()), static_cast<int>(w.rows()),
            std::vector<float>(w.data(), w.data()+w.size()), std::vector<float>(b.data(), b.data()+b.size())};
    };
    const auto dense = [&](const Linear& l) { return copy(l.weight, l.bias); };
    result->hidden = impl_->hidden; result->embedding = impl_->embedding;
    result->components = impl_->components; result->sigmaFloor = impl_->sigmaFloor;
    result->encoder0 = dense(impl_->encoder0); result->encoder2 = dense(impl_->encoder2);
    result->initial0 = dense(impl_->initial.first); result->initial2 = dense(impl_->initial.second);
    result->initial4 = dense(impl_->initial.last);
    result->hazard0 = dense(impl_->hazard.first); result->hazard2 = dense(impl_->hazard.second);
    result->hazard4 = dense(impl_->hazard.last);
    result->mixture0 = dense(impl_->mixture.first); result->mixture2 = dense(impl_->mixture.second);
    result->mixture4 = dense(impl_->mixture.last);
    result->gruInput = copy(impl_->weightInput, impl_->biasInput);
    result->gruHidden = copy(impl_->weightHidden, impl_->biasHidden);
    return result;
}

} // namespace mf

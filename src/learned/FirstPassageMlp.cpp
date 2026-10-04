#include "macrofacet/learned/FirstPassageMlp.h"
#include <nlohmann/json.hpp>
#include <torch/cuda.h>
#include <torch/script.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mf {
namespace {

using Json = nlohmann::json;

Vector3 rangeEndpoint(const Json& range, int endpoint) {
    const std::array<const char*, 3> names{"beta_0", "beta_a", "beta_g"};
    Vector3 result;
    for (int index = 0; index < 3; ++index) {
        const Json& values = range.at(names[static_cast<std::size_t>(index)]);
        if (!values.is_array() || values.size() != 2) {
            throw std::invalid_argument(
                std::string("model valid_range.") + names[static_cast<std::size_t>(index)] +
                " must contain [minimum, maximum]");
        }
        result[index] = values.at(endpoint).get<double>();
    }
    return result;
}

FirstPassageMlpMetadata loadMetadata(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open first-passage model bundle: " + path.string());
    Json root;
    stream >> root;
    FirstPassageMlpMetadata result;
    result.format = root.at("format").get<std::string>();
    result.formatVersion = root.at("format_version").get<int>();
    if (result.format != "macrofacet.fpt-ispline" || result.formatVersion != 1) {
        throw std::invalid_argument("unsupported first-passage model bundle format");
    }
    result.kernelId = root.at("kernel_id").get<std::string>();
    result.kernelType = root.at("kernel_type").get<std::string>();
    const Json& basis = root.at("basis");
    result.basisDegree = basis.at("degree").get<int>();
    result.basisCount = basis.at("basis_count").get<int>();
    result.maximumQ = basis.at("q_max").get<double>();
    result.knots = basis.at("knots").get<std::vector<double>>();
    result.validMinimum = rangeEndpoint(root.at("valid_range"), 0);
    result.validMaximum = rangeEndpoint(root.at("valid_range"), 1);

    if (result.kernelId.empty() || result.kernelType.empty() || result.basisDegree < 0 ||
        result.basisCount < 2 || !(result.maximumQ > 0.0) ||
        !std::isfinite(result.maximumQ) ||
        result.knots.size() != static_cast<std::size_t>(
            result.basisCount + result.basisDegree + 1) ||
        !std::is_sorted(result.knots.begin(), result.knots.end()) ||
        result.knots.front() != 0.0 || result.knots.back() != result.maximumQ ||
        !result.validMinimum.allFinite() || !result.validMaximum.allFinite() ||
        (result.validMinimum.array() > result.validMaximum.array()).any()) {
        throw std::invalid_argument("invalid first-passage model metadata");
    }
    return result;
}

double dot(const std::vector<double>& first, const std::vector<double>& second) {
    if (first.size() != second.size()) {
        throw std::invalid_argument("coefficient and spline basis sizes differ");
    }
    double result = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!(first[index] >= 0.0) || !std::isfinite(first[index]) ||
            !std::isfinite(second[index])) {
            throw std::invalid_argument("non-finite or negative first-passage coefficient");
        }
        result += first[index] * second[index];
    }
    if (!(result >= 0.0) || !std::isfinite(result)) {
        throw std::runtime_error("first-passage spline evaluation is invalid");
    }
    return result;
}

} // namespace

struct FirstPassageMlp::Impl {
    torch::Device device = torch::kCPU;
    torch::jit::Module module;
};

FirstPassageMlp::FirstPassageMlp(FirstPassageMlpConfig config)
    : config_(std::move(config)), metadata_(loadMetadata(config_.bundlePath)),
      impl_(std::make_unique<Impl>()) {
    if (config_.type != "mlp") {
        throw std::invalid_argument("first-passage model type must be 'mlp'");
    }
    if (config_.modulePath.empty() || config_.bundlePath.empty()) {
        throw std::invalid_argument("first-passage MLP requires module and bundle paths");
    }
    if (config_.device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else if (config_.device == "cuda") {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("first-passage MLP requested CUDA but CUDA is unavailable");
        }
        impl_->device = torch::Device(torch::kCUDA);
    } else {
        throw std::invalid_argument("first-passage MLP device must be 'cpu' or 'cuda'");
    }
    try {
        impl_->module = torch::jit::load(config_.modulePath.string(), impl_->device);
        impl_->module.eval();
    } catch (const c10::Error& error) {
        throw std::runtime_error(
            "cannot load first-passage TorchScript module '" +
            config_.modulePath.string() + "': " + error.what_without_backtrace());
    }
}

FirstPassageMlp::~FirstPassageMlp() = default;
FirstPassageMlp::FirstPassageMlp(FirstPassageMlp&&) noexcept = default;
FirstPassageMlp& FirstPassageMlp::operator=(FirstPassageMlp&&) noexcept = default;

std::vector<double> FirstPassageMlp::coefficients(const Vector3& beta) const {
    if (!beta.allFinite() || !(beta.z() > 0.0)) {
        throw std::invalid_argument("first-passage MLP beta must be finite with beta_g > 0");
    }
    const std::array<float, 3> values{
        static_cast<float>(beta.x()), static_cast<float>(beta.y()),
        static_cast<float>(beta.z())};
    torch::InferenceMode inference;
    torch::Tensor input = torch::from_blob(
        const_cast<float*>(values.data()), {1, 3}, torch::TensorOptions().dtype(torch::kFloat32))
        .clone().to(impl_->device);
    torch::Tensor output;
    try {
        output = impl_->module.forward({input}).toTensor();
    } catch (const c10::Error& error) {
        throw std::runtime_error(
            std::string("first-passage TorchScript inference failed: ") +
            error.what_without_backtrace());
    }
    if (output.dim() == 2 && output.size(0) == 1) output = output.squeeze(0);
    if (output.dim() != 1 || output.size(0) != metadata_.basisCount) {
        throw std::runtime_error("first-passage MLP returned an unexpected coefficient shape");
    }
    output = output.to(torch::kCPU, torch::kFloat64).contiguous();
    const double* data = output.data_ptr<double>();
    std::vector<double> result(data, data + metadata_.basisCount);
    for (double value : result) {
        if (!(value >= 0.0) || !std::isfinite(value)) {
            throw std::runtime_error("first-passage MLP returned an invalid coefficient");
        }
    }
    return result;
}

bool FirstPassageMlp::contains(const Vector3& beta, double q) const {
    return beta.allFinite() && std::isfinite(q) && q >= 0.0 && q <= metadata_.maximumQ &&
           (beta.array() >= metadata_.validMinimum.array()).all() &&
           (beta.array() <= metadata_.validMaximum.array()).all();
}

std::vector<double> FirstPassageMlp::mSpline(double q) const {
    std::vector<double> result(static_cast<std::size_t>(metadata_.basisCount), 0.0);
    if (!std::isfinite(q)) throw std::invalid_argument("spline coordinate must be finite");
    if (q < 0.0 || q > metadata_.maximumQ) return result;
    const int degree = metadata_.basisDegree;
    const std::vector<double>& knots = metadata_.knots;

    std::vector<double> level(knots.size() - 1, 0.0);
    if (q == metadata_.maximumQ) {
        result.back() = static_cast<double>(degree + 1) /
            (knots[static_cast<std::size_t>(metadata_.basisCount + degree)] -
             knots[static_cast<std::size_t>(metadata_.basisCount - 1)]);
        return result;
    }
    for (std::size_t index = 0; index + 1 < knots.size(); ++index) {
        if (q >= knots[index] && q < knots[index + 1]) level[index] = 1.0;
    }
    for (int order = 1; order <= degree; ++order) {
        std::vector<double> next(knots.size() - static_cast<std::size_t>(order) - 1, 0.0);
        for (std::size_t index = 0; index < next.size(); ++index) {
            const double leftWidth = knots[index + static_cast<std::size_t>(order)] - knots[index];
            const double rightWidth = knots[index + static_cast<std::size_t>(order) + 1] -
                                      knots[index + 1];
            if (leftWidth > 0.0) next[index] += (q - knots[index]) / leftWidth * level[index];
            if (rightWidth > 0.0) {
                next[index] +=
                    (knots[index + static_cast<std::size_t>(order) + 1] - q) /
                    rightWidth * level[index + 1];
            }
        }
        level = std::move(next);
    }
    for (int index = 0; index < metadata_.basisCount; ++index) {
        const double width = knots[static_cast<std::size_t>(index + degree + 1)] -
                             knots[static_cast<std::size_t>(index)];
        if (!(width > 0.0)) throw std::runtime_error("M-spline has zero support width");
        result[static_cast<std::size_t>(index)] =
            static_cast<double>(degree + 1) / width * level[static_cast<std::size_t>(index)];
    }
    return result;
}

std::vector<double> FirstPassageMlp::iSpline(double q) const {
    std::vector<double> result(static_cast<std::size_t>(metadata_.basisCount), 0.0);
    if (!std::isfinite(q)) throw std::invalid_argument("spline coordinate must be finite");
    if (q <= 0.0) return result;
    if (q >= metadata_.maximumQ) {
        std::fill(result.begin(), result.end(), 1.0);
        return result;
    }

    // Eight-point Gauss-Legendre integrates every polynomial basis used by the
    // current bundles exactly up to floating-point roundoff. Splitting at each
    // knot avoids integrating across a piecewise-polynomial discontinuity.
    constexpr std::array<double, 8> nodes{
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
        -0.1834346424956498,  0.1834346424956498,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363};
    constexpr std::array<double, 8> weights{
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
        0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763};
    for (std::size_t interval = 0; interval + 1 < metadata_.knots.size(); ++interval) {
        const double lo = metadata_.knots[interval];
        const double hi = std::min(q, metadata_.knots[interval + 1]);
        if (!(hi > lo)) continue;
        const double center = 0.5 * (lo + hi);
        const double radius = 0.5 * (hi - lo);
        for (std::size_t sample = 0; sample < nodes.size(); ++sample) {
            const std::vector<double> m = mSpline(center + radius * nodes[sample]);
            const double scale = radius * weights[sample];
            for (std::size_t basis = 0; basis < result.size(); ++basis) {
                result[basis] += scale * m[basis];
            }
        }
        if (metadata_.knots[interval + 1] >= q) break;
    }
    for (double& value : result) value = std::clamp(value, 0.0, 1.0);
    return result;
}

double FirstPassageMlp::cumulativeHazard(
    const std::vector<double>& coefficientsValue, double q) const {
    return dot(coefficientsValue, iSpline(q));
}

double FirstPassageMlp::transmittance(
    const std::vector<double>& coefficientsValue, double q) const {
    return std::exp(-cumulativeHazard(coefficientsValue, q));
}

double FirstPassageMlp::dimensionlessExtinction(
    const std::vector<double>& coefficientsValue, double q) const {
    return dot(coefficientsValue, mSpline(q));
}

double FirstPassageMlp::inverseCumulativeHazard(
    const std::vector<double>& coefficientsValue, double opticalDepth, int iterations) const {
    if (!(opticalDepth >= 0.0) || !std::isfinite(opticalDepth) || iterations < 1) {
        throw std::invalid_argument("invalid optical depth inversion request");
    }
    if (opticalDepth == 0.0) return 0.0;
    const double maximum = cumulativeHazard(coefficientsValue, metadata_.maximumQ);
    if (opticalDepth > maximum) return std::numeric_limits<double>::infinity();
    double lo = 0.0;
    double hi = metadata_.maximumQ;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const double middle = 0.5 * (lo + hi);
        if (cumulativeHazard(coefficientsValue, middle) < opticalDepth) lo = middle;
        else hi = middle;
    }
    return 0.5 * (lo + hi);
}

} // namespace mf

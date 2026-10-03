#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/FirstPassageRice.h"

#include "macrofacet/core/Random.h"
#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <Eigen/Cholesky>
#include <unsupported/Eigen/FFT>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace mf {
namespace {

using Json = nlohmann::json;

void requireFinite(double value, const std::string& name) {
    if (!std::isfinite(value)) throw std::invalid_argument(name + " must be finite");
}

void requirePositive(double value, const std::string& name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(name + " must be a finite positive number");
    }
}

bool closeInteger(double value) {
    return std::abs(value - std::round(value)) <=
           1e-9 * std::max(1.0, std::abs(value));
}

std::string csv(const std::string& value) {
    std::string result = "\"";
    for (const char ch : value) {
        if (ch == '\"') result += '\"';
        result += ch;
    }
    result += '\"';
    return result;
}

std::string xml(const std::string& value) {
    std::string result;
    for (const char ch : value) {
        if (ch == '&') result += "&amp;";
        else if (ch == '<') result += "&lt;";
        else if (ch == '>') result += "&gt;";
        else if (ch == '\"') result += "&quot;";
        else result += ch;
    }
    return result;
}

std::ofstream openOutput(const std::filesystem::path& path) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write output: " + path.string());
    stream << std::setprecision(17);
    return stream;
}

using StationaryKernel = FirstPassageStationaryKernel;

std::size_t nextPowerOfTwo(std::size_t value) {
    std::size_t result = 1;
    while (result < value) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error("first-passage grid is too large");
        }
        result *= 2;
    }
    return result;
}

class CirculantSampler {
public:
    CirculantSampler(const StationaryKernel& kernel, std::size_t sampleCount,
                     double step, int maximumExpansions)
        : kernel_(kernel), sampleCount_(sampleCount), step_(step) {
        if (sampleCount_ < 2) throw std::invalid_argument("first-passage grid is empty");
        embeddingSize_ = nextPowerOfTwo(2 * (sampleCount_ - 1));
        for (int expansion = 0; expansion <= maximumExpansions; ++expansion) {
            std::vector<std::complex<double>> column(embeddingSize_);
            for (std::size_t i = 0; i < embeddingSize_; ++i) {
                const std::size_t lag = std::min(i, embeddingSize_ - i);
                column[i] = {kernel_.covariance(static_cast<double>(lag) * step_), 0.0};
            }
            fft_.fwd(frequencies_, column);
            double minimum = std::numeric_limits<double>::infinity();
            double maximumImaginary = 0.0;
            for (const auto& value : frequencies_) {
                minimum = std::min(minimum, value.real());
                maximumImaginary = std::max(maximumImaginary, std::abs(value.imag()));
            }
            const double scale = kernel_.config().variance;
            const double tolerance = 1e-10 * std::max(1.0, scale);
            if (maximumImaginary <= tolerance && minimum >= -tolerance) {
                minimumEigenvalue_ = minimum;
                squareRoots_.resize(embeddingSize_);
                for (std::size_t i = 0; i < embeddingSize_; ++i) {
                    squareRoots_[i] = std::sqrt(std::max(0.0, frequencies_[i].real()));
                }
                time_.resize(embeddingSize_);
                return;
            }
            if (expansion == maximumExpansions) break;
            embeddingSize_ *= 2;
        }
        throw std::runtime_error(
            "circulant embedding remained indefinite for kernel " +
            kernel_.config().id + "; increase sampler.max_embedding_expansions");
    }

    std::vector<double> sampleConditionedValue(Random& rng, double mean,
                                                 double initialValue) {
        drawZeroMean(rng);
        std::vector<double> result(sampleCount_);
        const double innovation = initialValue - mean - time_[0].real();
        const double variance = kernel_.config().variance;
        for (std::size_t i = 0; i < sampleCount_; ++i) {
            const double t = static_cast<double>(i) * step_;
            result[i] = mean + time_[i].real() +
                        kernel_.covariance(t) / variance * innovation;
        }
        // Remove the final FFT roundoff at the exactly conditioned site.
        result[0] = initialValue;
        return result;
    }

    std::vector<double> sampleConditionedValueDerivative(
        Random& rng, double initialValue, double initialDerivative) {
        drawZeroMean(rng);
        const double valueSample = time_[0].real();
        const double derivativeSample =
            (time_[1].real() - time_[embeddingSize_ - 1].real()) / (2.0 * step_);
        const double variance = kernel_.config().variance;
        const double derivativeVariance =
            (variance - circulantCovariance(1, embeddingSize_ - 1)) /
            (2.0 * step_ * step_);
        if (!(derivativeVariance > 0.0) || !std::isfinite(derivativeVariance)) {
            throw std::runtime_error("finite-difference derivative observation is degenerate");
        }
        std::vector<double> result(sampleCount_);
        for (std::size_t i = 0; i < sampleCount_; ++i) {
            const double valueCovariance = circulantCovariance(i, 0);
            const double derivativeCovariance =
                (circulantCovariance(i, 1) -
                 circulantCovariance(i, embeddingSize_ - 1)) / (2.0 * step_);
            result[i] = time_[i].real() +
                valueCovariance / variance * (initialValue - valueSample) +
                derivativeCovariance / derivativeVariance *
                    (initialDerivative - derivativeSample);
        }
        result[0] = initialValue;
        return result;
    }

    std::size_t embeddingSize() const { return embeddingSize_; }
    double minimumEigenvalue() const { return minimumEigenvalue_; }

private:
    void drawZeroMean(Random& rng) {
        // Draw conjugate-symmetric Fourier coefficients directly. Compared with
        // filtering real white noise, this has the same circulant covariance but
        // needs only the inverse FFT. Eigen's inverse carries a 1/m factor, hence
        // the sqrt(m) scale below.
        const double rootSize = std::sqrt(static_cast<double>(embeddingSize_));
        frequencies_[0] = {rootSize * squareRoots_[0] * rng.standardNormal(), 0.0};
        const std::size_t half = embeddingSize_ / 2;
        frequencies_[half] = {
            rootSize * squareRoots_[half] * rng.standardNormal(), 0.0};
        for (std::size_t i = 1; i < half; ++i) {
            const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
            const std::complex<double> normal(
                rng.standardNormal() * inverseSqrtTwo,
                rng.standardNormal() * inverseSqrtTwo);
            frequencies_[i] = rootSize * squareRoots_[i] * normal;
            frequencies_[embeddingSize_ - i] = std::conj(frequencies_[i]);
        }
        fft_.inv(time_, frequencies_);
    }

    double circulantCovariance(std::size_t first, std::size_t second) const {
        const std::size_t direct = first >= second ? first - second : second - first;
        const std::size_t lag = std::min(direct, embeddingSize_ - direct);
        return kernel_.covariance(static_cast<double>(lag) * step_);
    }

    const StationaryKernel& kernel_;
    std::size_t sampleCount_ = 0;
    double step_ = 0.0;
    std::size_t embeddingSize_ = 0;
    double minimumEigenvalue_ = 0.0;
    Eigen::FFT<double> fft_;
    std::vector<std::complex<double>> frequencies_;
    std::vector<double> squareRoots_;
    std::vector<std::complex<double>> time_;
};

double crossingTime(const std::vector<double>& path, std::size_t stride,
                    double fineStep, double threshold, double maximumTime) {
    double previous = path[0];
    for (std::size_t index = stride; index < path.size(); index += stride) {
        const double current = path[index];
        if (current <= threshold) {
            const double left = static_cast<double>(index - stride) * fineStep;
            const double width = static_cast<double>(stride) * fineStep;
            const double denominator = previous - current;
            const double fraction = denominator > 0.0
                ? std::clamp((previous - threshold) / denominator, 0.0, 1.0)
                : 1.0;
            return std::min(maximumTime, left + fraction * width);
        }
        previous = current;
    }
    return std::numeric_limits<double>::infinity();
}

using State2 = Eigen::Vector2d;
using Matrix2 = Eigen::Matrix2d;

struct StateTransition2 {
    Matrix2 matrix = Matrix2::Identity();
    Matrix2 covariance = Matrix2::Zero();
};

StateTransition2 matern32Transition(double step) {
    requirePositive(step, "Matérn 3/2 state transition step");
    constexpr double lambda = 1.7320508075688772935; // sqrt(3), q=t/ell
    const double r = lambda * step;
    const double exponential = std::exp(-r);
    StateTransition2 result;
    result.matrix << exponential * (1.0 + r), exponential * step,
                     -exponential * lambda * r, exponential * (1.0 - r);

    const double x = 2.0 * r;
    double q00 = 0.0;
    if (x < 1.0) {
        // 1-exp(-x)(1+x+x^2/2), evaluated without the O(x^3)
        // cancellation that would otherwise dominate at refinement scales.
        double term = x * x * x / 6.0;
        double tail = term;
        for (int order = 4; order < 64; ++order) {
            term *= x / static_cast<double>(order);
            tail += term;
            if (term <= 2e-16 * std::max(1.0, tail)) break;
        }
        q00 = std::exp(-x) * tail;
    } else {
        q00 = 1.0 - std::exp(-x) * (1.0 + x + 0.5 * x * x);
    }
    const double q01 = 2.0 * lambda * r * r * std::exp(-2.0 * r);
    const double q11 = lambda * lambda *
        (-std::expm1(-2.0 * r) + std::exp(-2.0 * r) * (2.0 * r - 2.0 * r * r));
    result.covariance << q00, q01, q01, q11;
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

State2 sampleGaussian2(const State2& mean, const Matrix2& source, Random& rng) {
    Matrix2 covariance = 0.5 * (source + source.transpose());
    const double scale = std::max({std::abs(covariance(0, 0)),
                                  std::abs(covariance(0, 1)),
                                  std::abs(covariance(1, 1)),
                                  std::numeric_limits<double>::min()});
    const double tolerance = 256.0 * std::numeric_limits<double>::epsilon() * scale;
    if (covariance(0, 0) < -tolerance || covariance(1, 1) < -tolerance) {
        throw std::runtime_error("negative Matérn state covariance diagonal");
    }
    covariance(0, 0) = std::max(0.0, covariance(0, 0));
    covariance(1, 1) = std::max(0.0, covariance(1, 1));
    State2 sample = mean;
    if (covariance(0, 0) > 0.0) {
        const double l00 = std::sqrt(covariance(0, 0));
        const double l10 = covariance(1, 0) / l00;
        double remainder = covariance(1, 1) - l10 * l10;
        if (remainder < -tolerance) {
            throw std::runtime_error("Matérn state covariance is not PSD");
        }
        remainder = std::max(0.0, remainder);
        const double z0 = rng.standardNormal();
        const double z1 = rng.standardNormal();
        sample[0] += l00 * z0;
        sample[1] += l10 * z0 + std::sqrt(remainder) * z1;
    } else {
        if (std::abs(covariance(0, 1)) > tolerance) {
            throw std::runtime_error("singular Matérn covariance has nonzero cross term");
        }
        sample[1] += std::sqrt(covariance(1, 1)) * rng.standardNormal();
    }
    return sample;
}

State2 advanceMatern32(const State2& state, double step, Random& rng) {
    const StateTransition2 transition = matern32Transition(step);
    return sampleGaussian2(transition.matrix * state, transition.covariance, rng);
}

struct BridgeDistribution2 {
    State2 mean = State2::Zero();
    Matrix2 covariance = Matrix2::Zero();
};

BridgeDistribution2 matern32BridgeMidpoint(const State2& left, const State2& right,
                                            double width) {
    const double half = 0.5 * width;
    const StateTransition2 first = matern32Transition(half);
    const StateTransition2 second = matern32Transition(width - half);
    const StateTransition2 total = matern32Transition(width);
    const Matrix2 cross = first.covariance * second.matrix.transpose();
    Eigen::LDLT<Matrix2> factor(total.covariance);
    if (factor.info() != Eigen::Success) {
        throw std::runtime_error("Matérn bridge covariance factorization failed");
    }
    const Matrix2 gain = factor.solve(cross.transpose()).transpose();
    BridgeDistribution2 result;
    result.mean = first.matrix * left +
        gain * (right - total.matrix * left);
    result.covariance = first.covariance - gain * cross.transpose();
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

struct HermitePolynomial {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;

    double value(double s) const { return ((a * s + b) * s + c) * s + d; }
    double derivative(double s) const { return (3.0 * a * s + 2.0 * b) * s + c; }
};

HermitePolynomial hermitePolynomial(double value0, double derivative0,
                                    double value1, double derivative1, double width) {
    return {2.0 * value0 - 2.0 * value1 + width * (derivative0 + derivative1),
            -3.0 * value0 + 3.0 * value1 - width * (2.0 * derivative0 + derivative1),
            width * derivative0, value0};
}

std::vector<double> hermiteKnots(const HermitePolynomial& polynomial) {
    std::vector<double> result{0.0, 1.0};
    const double qa = 3.0 * polynomial.a;
    const double qb = 2.0 * polynomial.b;
    const double qc = polynomial.c;
    const double scale = std::max({1.0, std::abs(qa), std::abs(qb), std::abs(qc)});
    if (std::abs(qa) <= 1e-14 * scale) {
        if (std::abs(qb) > 1e-14 * scale) {
            const double root = -qc / qb;
            if (root > 0.0 && root < 1.0) result.push_back(root);
        }
    } else {
        const double discriminant = qb * qb - 4.0 * qa * qc;
        if (discriminant >= 0.0) {
            const double root = std::sqrt(std::max(0.0, discriminant));
            const double r0 = (-qb - root) / (2.0 * qa);
            const double r1 = (-qb + root) / (2.0 * qa);
            if (r0 > 0.0 && r0 < 1.0) result.push_back(r0);
            if (r1 > 0.0 && r1 < 1.0) result.push_back(r1);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end(), [](double a, double b) {
        return std::abs(a - b) <= 1e-14;
    }), result.end());
    return result;
}

double hermiteMinimum(const HermitePolynomial& polynomial) {
    double result = std::numeric_limits<double>::infinity();
    for (double s : hermiteKnots(polynomial)) result = std::min(result, polynomial.value(s));
    return result;
}

struct HermiteCrossing {
    bool found = false;
    double fraction = 0.0;
    double derivative = 0.0;
};

HermiteCrossing firstHermiteDowncrossing(const HermitePolynomial& polynomial,
                                         double width, double tolerance) {
    const std::vector<double> knots = hermiteKnots(polynomial);
    for (std::size_t i = 1; i < knots.size(); ++i) {
        double left = knots[i - 1];
        double right = knots[i];
        const double leftValue = polynomial.value(left);
        const double rightValue = polynomial.value(right);
        // At q=0 the process starts on the boundary with positive derivative;
        // that prescribed birth point is not itself a new first passage.
        if (!(leftValue > 0.0 || (left == 0.0 && polynomial.derivative(0.0) > 0.0)) ||
            rightValue > 0.0) {
            continue;
        }
        double lo = left;
        double hi = right;
        for (int iteration = 0; iteration < 80 && (hi - lo) * width > tolerance; ++iteration) {
            const double middle = 0.5 * (lo + hi);
            if (polynomial.value(middle) > 0.0) lo = middle;
            else hi = middle;
        }
        const double root = 0.5 * (lo + hi);
        const double derivative = polynomial.derivative(root) / width;
        if (derivative <= 1e-10) return {true, root, derivative};
    }
    return {};
}

struct CollisionSample {
    double eventQ = 0.0;
    bool event = false;
    double crossingSlope = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t seed = 0;
    std::uint64_t transitions = 0;
    std::uint64_t bridgeRefinements = 0;
    int deepestRefinement = 0;
};

double collisionField(const FirstPassageCollisionState& parameter, double q,
                      const State2& state) {
    return parameter.beta0 + parameter.betaMeanSlope * q + state[0];
}

double collisionDerivative(const FirstPassageCollisionState& parameter,
                           const State2& state) {
    return parameter.betaMeanSlope + state[1];
}

struct IntervalCrossing {
    bool found = false;
    double q = 0.0;
    double slope = 0.0;
};

IntervalCrossing inspectCollisionInterval(
    const FirstPassageCollisionState& parameter,
    const FirstPassageCollisionSamplerConfig& settings,
    double q0, const State2& state0, double q1, const State2& state1,
    int depth, Random& rng, CollisionSample& diagnostics) {
    diagnostics.deepestRefinement = std::max(diagnostics.deepestRefinement, depth);
    const double width = q1 - q0;
    const double value0 = collisionField(parameter, q0, state0);
    const double value1 = collisionField(parameter, q1, state1);
    const double derivative0 = collisionDerivative(parameter, state0);
    const double derivative1 = collisionDerivative(parameter, state1);
    const HermitePolynomial polynomial = hermitePolynomial(
        value0, derivative0, value1, derivative1, width);
    const HermiteCrossing coarse = firstHermiteDowncrossing(
        polynomial, width, settings.crossingTolerance);

    if (width <= settings.minimumStep || depth >= settings.maximumRefinementDepth) {
        if (!coarse.found) return {};
        return {true, q0 + coarse.fraction * width, coarse.derivative};
    }

    const BridgeDistribution2 bridge = matern32BridgeMidpoint(state0, state1, width);
    const double midpointQ = 0.5 * (q0 + q1);
    const double midpointMean = parameter.beta0 +
        parameter.betaMeanSlope * midpointQ + bridge.mean[0];
    const double midpointStddev = std::sqrt(std::max(0.0, bridge.covariance(0, 0)));
    const bool suspicious = coarse.found ||
        hermiteMinimum(polynomial) <= settings.bridgeSigmaMargin * midpointStddev ||
        midpointMean <= settings.bridgeSigmaMargin * midpointStddev;
    if (!suspicious) return {};

    ++diagnostics.bridgeRefinements;
    const State2 midpoint = sampleGaussian2(bridge.mean, bridge.covariance, rng);
    const IntervalCrossing left = inspectCollisionInterval(
        parameter, settings, q0, state0, midpointQ, midpoint,
        depth + 1, rng, diagnostics);
    if (left.found) return left;
    return inspectCollisionInterval(parameter, settings, midpointQ, midpoint, q1, state1,
                                    depth + 1, rng, diagnostics);
}

std::uint64_t mixSeed(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::uint64_t collisionSeed(std::uint64_t root, std::size_t kernel,
                            std::size_t state, std::size_t resolution,
                            std::uint64_t trajectory) {
    std::uint64_t value = mixSeed(root ^ 0x636f6c6c6973696fULL);
    // The dimensionless process is independent of sigma and ell. Reusing the
    // same stream across physical kernel scales makes that invariance directly
    // testable instead of burying it under independent Monte Carlo noise.
    (void)kernel;
    value ^= mixSeed(static_cast<std::uint64_t>(state) + 0x10001ULL);
    value ^= mixSeed(static_cast<std::uint64_t>(resolution) + 0x20001ULL);
    value ^= mixSeed(trajectory + 0x30001ULL);
    return mixSeed(value);
}

CollisionSample sampleCollisionFirstPassage(
    const FirstPassageCollisionState& parameter,
    const FirstPassageCollisionSamplerConfig& settings,
    double maximumQ, double baseStep, std::uint64_t seed) {
    CollisionSample result;
    result.eventQ = maximumQ;
    result.seed = seed;
    Random rng(seed);
    State2 state(-parameter.beta0,
                 parameter.betaCollisionSlope - parameter.betaMeanSlope);
    double q = 0.0;
    while (q < maximumQ) {
        const double step = std::min(baseStep, maximumQ - q);
        const State2 next = advanceMatern32(state, step, rng);
        ++result.transitions;
        const IntervalCrossing crossing = inspectCollisionInterval(
            parameter, settings, q, state, q + step, next, 0, rng, result);
        if (crossing.found) {
            result.event = true;
            result.eventQ = crossing.q;
            result.crossingSlope = std::max(0.0, -crossing.slope);
            return result;
        }
        q += step;
        state = next;
    }
    return result;
}

CollisionSample sampleGridCollisionFirstPassage(
    CirculantSampler& sampler, const FirstPassageCollisionState& parameter,
    double maximumQ, double step, std::uint64_t seed) {
    CollisionSample result;
    result.eventQ = maximumQ;
    result.seed = seed;
    Random rng(seed);
    const std::vector<double> residual = sampler.sampleConditionedValueDerivative(
        rng, -parameter.beta0,
        parameter.betaCollisionSlope - parameter.betaMeanSlope);
    result.transitions = residual.size() - 1;
    std::vector<double> values(residual.size());
    values[0] = 0.0;
    for (std::size_t index = 1; index < residual.size(); ++index) {
        const double q = static_cast<double>(index) * step;
        values[index] = parameter.beta0 + parameter.betaMeanSlope * q +
                        residual[index];
    }
    for (std::size_t index = 1; index < values.size(); ++index) {
        const double derivative0 = index == 1
            ? parameter.betaCollisionSlope
            : (values[index] - values[index - 2]) / (2.0 * step);
        const double derivative1 = index + 1 < values.size()
            ? (values[index + 1] - values[index - 1]) / (2.0 * step)
            : (values[index] - values[index - 1]) / step;
        const HermitePolynomial polynomial = hermitePolynomial(
            values[index - 1], derivative0, values[index], derivative1, step);
        const HermiteCrossing crossing = firstHermiteDowncrossing(
            polynomial, step, 1e-12 * step);
        if (crossing.found) {
            result.event = true;
            result.eventQ = (static_cast<double>(index - 1) + crossing.fraction) * step;
            result.crossingSlope = std::max(0.0, -crossing.derivative);
            return result;
        }
    }
    return result;
}

struct ProportionInterval {
    double lower = 0.0;
    double upper = 0.0;
};

ProportionInterval wilsonInterval(std::uint64_t successes, std::uint64_t total,
                                  double z) {
    if (total == 0) return {};
    const double n = static_cast<double>(total);
    const double p = static_cast<double>(successes) / n;
    const double z2 = z * z;
    const double denominator = 1.0 + z2 / n;
    const double center = (p + z2 / (2.0 * n)) / denominator;
    const double half = z / denominator *
        std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n));
    return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

double expectedNegativePart(double mean, double stddev) {
    if (!(stddev > 0.0)) return std::max(0.0, -mean);
    const double z = mean / stddev;
    return stddev * normalPdf(z) - mean * normalCdf(-z);
}

double pointwiseHazard(const StationaryKernel& kernel, double mean,
                       double threshold) {
    const double valueStddev = std::sqrt(kernel.config().variance);
    const double z = (threshold - mean) / valueStddev;
    const double positiveProbability = normalSurvival(z);
    if (!(positiveProbability > 0.0)) return std::numeric_limits<double>::infinity();
    const double density = normalPdf(z) / valueStddev;
    const double projectedSpeed = std::sqrt(kernel.derivativeVariance()) * kInvSqrtTwoPi;
    return density * projectedSpeed / positiveProbability;
}

double endpointConditionedHazard(const StationaryKernel& kernel, double mean,
                                 double threshold, double initialValue,
                                 double t) {
    if (!(t > 0.0)) return 0.0;
    const double variance = kernel.config().variance;
    const double covariance = kernel.covariance(t);
    const double covarianceDerivative = kernel.firstDerivative(t);
    const double delta = initialValue - mean;
    const double valueMean = mean + covariance / variance * delta;
    const double derivativeMean = covarianceDerivative / variance * delta;
    const double valueVariance = variance - covariance * covariance / variance;
    const double derivativeVariance = kernel.derivativeVariance() -
        covarianceDerivative * covarianceDerivative / variance;
    const double valueDerivativeCovariance =
        -covariance * covarianceDerivative / variance;
    if (!(valueVariance > 1e-15 * variance)) return 0.0;
    const double valueStddev = std::sqrt(std::max(0.0, valueVariance));
    const double z = (threshold - valueMean) / valueStddev;
    const double survival = normalSurvival(z);
    if (!(survival > 0.0)) return std::numeric_limits<double>::infinity();
    const double boundaryDensity = normalPdf(z) / valueStddev;
    const double speedMean = derivativeMean +
        valueDerivativeCovariance / valueVariance * (threshold - valueMean);
    const double speedVariance = std::max(0.0, derivativeVariance -
        valueDerivativeCovariance * valueDerivativeCovariance / valueVariance);
    return boundaryDensity / survival *
           expectedNegativePart(speedMean, std::sqrt(speedVariance));
}

double startConditionedEndpointSurvival(const StationaryKernel& kernel, double mean,
                                        double threshold, double initialValue,
                                        double t) {
    if (!(t > 0.0)) return initialValue > threshold ? 1.0 : 0.0;
    const double variance = kernel.config().variance;
    const double covariance = kernel.covariance(t);
    const double conditionalMean = mean +
        covariance / variance * (initialValue - mean);
    const double conditionalVariance = std::max(
        0.0, variance - covariance * covariance / variance);
    if (!(conditionalVariance > 1e-15 * variance)) {
        return conditionalMean > threshold ? 1.0 : 0.0;
    }
    return normalSurvival(
        (threshold - conditionalMean) / std::sqrt(conditionalVariance));
}

struct CurvePoint {
    double begin = 0.0;
    double end = 0.0;
    double time = 0.0;
    std::uint64_t atRisk = 0;
    std::uint64_t events = 0;
    double survival = 0.0;
    double survivalLower = 0.0;
    double survivalUpper = 0.0;
    double density = 0.0;
    double hazard = 0.0;
    double hazardLower = 0.0;
    double hazardUpper = 0.0;
    double cumulativeHazard = 0.0;
    double survivalFromHazard = 0.0;
    double endpointHazard = 0.0;
    double pointwiseHazard = 0.0;
    double startConditionedEndpointSurvival = 0.0;
    double startConditionedSigma1Survival = 0.0;
    double pointwiseSigma2Survival = 0.0;
    double riceW1 = std::numeric_limits<double>::quiet_NaN();
    double riceW2Correction = std::numeric_limits<double>::quiet_NaN();
    double riceW2AbsoluteError = std::numeric_limits<double>::quiet_NaN();
    bool riceW2Converged = false;
    double riceDensityOrder2 = std::numeric_limits<double>::quiet_NaN();
    double riceSurvivalOrder1 = std::numeric_limits<double>::quiet_NaN();
    double riceSurvivalOrder2 = std::numeric_limits<double>::quiet_NaN();
    double riceHazardOrder1 = std::numeric_limits<double>::quiet_NaN();
    double riceHazardOrder2 = std::numeric_limits<double>::quiet_NaN();
};

struct RiceCurvePoint {
    double w1 = std::numeric_limits<double>::quiet_NaN();
    double w2Correction = std::numeric_limits<double>::quiet_NaN();
    double w2AbsoluteError = std::numeric_limits<double>::quiet_NaN();
    bool w2Converged = false;
    double densityOrder2 = std::numeric_limits<double>::quiet_NaN();
    double survivalOrder1 = std::numeric_limits<double>::quiet_NaN();
    double survivalOrder2 = std::numeric_limits<double>::quiet_NaN();
    double hazardOrder1 = std::numeric_limits<double>::quiet_NaN();
    double hazardOrder2 = std::numeric_limits<double>::quiet_NaN();
};

std::vector<RiceCurvePoint> makeRiceCurve(
    const FirstPassageExperimentConfig& config,
    const StationaryKernel& kernel) {
    if (!config.riceSeries.enabled) return {};
    NumericPolicy policy = defaultNumericPolicy();
    policy.relativeTolerance = config.riceSeries.relativeTolerance;
    policy.absoluteTolerance = config.riceSeries.absoluteTolerance;
    policy.maxQuadratureSubdivisions =
        config.riceSeries.maximumQuadratureSubdivisions;
    policy.covarianceRoundoffMultiplier =
        config.riceSeries.covarianceRoundoffMultiplier;

    std::vector<RiceCurvePoint> result(
        static_cast<std::size_t>(config.curveBins));
    const double width = config.maximumTime / config.curveBins;
    double cumulativeOrder1 = 0.0;
    double cumulativeOrder2 = 0.0;
    for (int bin = 0; bin < config.curveBins; ++bin) {
        RiceCurvePoint& point = result[static_cast<std::size_t>(bin)];
        const double time = (static_cast<double>(bin) + 0.5) * width;
        try {
            point.w1 = riceDowncrossingW1(
                kernel, config.processMean, config.threshold,
                config.initialValue, time, policy);
        } catch (const NumericError& error) {
            throw NumericError(error.status(), "Rice W1 failed for kernel " +
                kernel.config().id + " at t=" + std::to_string(time) +
                ": " + error.what());
        }
        const double midSurvivalOrder1 =
            1.0 - cumulativeOrder1 - 0.5 * width * point.w1;
        point.hazardOrder1 = midSurvivalOrder1 > 0.0
            ? point.w1 / midSurvivalOrder1
            : std::numeric_limits<double>::quiet_NaN();
        cumulativeOrder1 += width * point.w1;
        point.survivalOrder1 = 1.0 - cumulativeOrder1;

        if (config.riceSeries.maxOrder >= 2) {
            IntegralResult correction;
            try {
                correction = riceSecondOrderCorrection(
                    kernel, config.processMean, config.threshold,
                    config.initialValue, time, policy);
            } catch (const NumericError& error) {
                throw NumericError(error.status(), "Rice W2 integral failed for kernel " +
                    kernel.config().id + " at t=" + std::to_string(time) +
                    ": " + error.what());
            }
            point.w2Correction = correction.value;
            point.w2AbsoluteError = correction.absError;
            point.w2Converged = correction.status == NumericStatus::Ok;
            point.densityOrder2 = point.w1 - point.w2Correction;
            const double midSurvivalOrder2 =
                1.0 - cumulativeOrder2 - 0.5 * width * point.densityOrder2;
            point.hazardOrder2 = midSurvivalOrder2 > 0.0
                ? point.densityOrder2 / midSurvivalOrder2
                : std::numeric_limits<double>::quiet_NaN();
            cumulativeOrder2 += width * point.densityOrder2;
            point.survivalOrder2 = 1.0 - cumulativeOrder2;
        }
    }
    return result;
}

struct CurveSeries {
    FirstPassageKernelConfig kernel;
    double step = 0.0;
    std::size_t embeddingSize = 0;
    double minimumEigenvalue = 0.0;
    std::vector<double> eventTimes;
    std::vector<CurvePoint> points;
    std::uint64_t events = 0;
    double restrictedMean = 0.0;
    double meanEventTime = std::numeric_limits<double>::quiet_NaN();
    double endpointRmse = std::numeric_limits<double>::quiet_NaN();
    double pointwiseRmse = std::numeric_limits<double>::quiet_NaN();
    double endpointNrmse = std::numeric_limits<double>::quiet_NaN();
    double pointwiseNrmse = std::numeric_limits<double>::quiet_NaN();
    double endpointSurvivalRmse = std::numeric_limits<double>::quiet_NaN();
    double sigma1SurvivalRmse = std::numeric_limits<double>::quiet_NaN();
    double sigma2SurvivalRmse = std::numeric_limits<double>::quiet_NaN();
};

std::vector<CurvePoint> makeCurve(const FirstPassageExperimentConfig& config,
                                  const StationaryKernel& kernel,
                                  const std::vector<double>& times,
                                  const std::vector<RiceCurvePoint>& riceCurve) {
    std::vector<CurvePoint> result;
    result.reserve(static_cast<std::size_t>(config.curveBins));
    const double width = config.maximumTime / config.curveBins;
    const double z = normalQuantile(0.5 + 0.5 * config.confidenceLevel);
    double cumulative = 0.0;
    double cumulativeEndpointHazard = 0.0;
    const double sigma2 = pointwiseHazard(
        kernel, config.processMean, config.threshold);
    for (int bin = 0; bin < config.curveBins; ++bin) {
        CurvePoint point;
        point.begin = static_cast<double>(bin) * width;
        point.end = static_cast<double>(bin + 1) * width;
        point.time = 0.5 * (point.begin + point.end);
        for (const double time : times) {
            if (time > point.begin) ++point.atRisk;
            if (time > point.begin && time <= point.end) ++point.events;
        }
        const std::uint64_t survivors = static_cast<std::uint64_t>(std::count_if(
            times.begin(), times.end(), [&](double time) { return time > point.end; }));
        point.survival = static_cast<double>(survivors) / times.size();
        const ProportionInterval survivalCi = wilsonInterval(survivors, times.size(), z);
        point.survivalLower = survivalCi.lower;
        point.survivalUpper = survivalCi.upper;
        point.density = static_cast<double>(point.events) /
                        (static_cast<double>(times.size()) * width);
        if (point.atRisk > 0) {
            const double probability = static_cast<double>(point.events) / point.atRisk;
            point.hazard = probability / width;
            const ProportionInterval hazardCi = wilsonInterval(point.events, point.atRisk, z);
            point.hazardLower = hazardCi.lower / width;
            point.hazardUpper = hazardCi.upper / width;
            cumulative += probability;
        }
        point.cumulativeHazard = cumulative;
        point.survivalFromHazard = std::exp(-cumulative);
        point.endpointHazard = endpointConditionedHazard(
            kernel, config.processMean, config.threshold, config.initialValue, point.time);
        point.pointwiseHazard = sigma2;
        cumulativeEndpointHazard += point.endpointHazard * width;
        point.startConditionedEndpointSurvival = startConditionedEndpointSurvival(
            kernel, config.processMean, config.threshold, config.initialValue, point.end);
        point.startConditionedSigma1Survival = std::exp(-cumulativeEndpointHazard);
        point.pointwiseSigma2Survival = std::exp(-sigma2 * point.end);
        if (!riceCurve.empty()) {
            const RiceCurvePoint& rice = riceCurve[static_cast<std::size_t>(bin)];
            point.riceW1 = rice.w1;
            point.riceW2Correction = rice.w2Correction;
            point.riceW2AbsoluteError = rice.w2AbsoluteError;
            point.riceW2Converged = rice.w2Converged;
            point.riceDensityOrder2 = rice.densityOrder2;
            point.riceSurvivalOrder1 = rice.survivalOrder1;
            point.riceSurvivalOrder2 = rice.survivalOrder2;
            point.riceHazardOrder1 = rice.hazardOrder1;
            point.riceHazardOrder2 = rice.hazardOrder2;
        }
        result.push_back(point);
    }
    return result;
}

struct StateRecord {
    std::uint64_t trajectory = 0;
    std::size_t snapshot = 0;
    double value = 0.0;
    double derivative = 0.0;
    bool eventInWindow = false;
};

struct KernelStateRecords {
    FirstPassageKernelConfig kernel;
    std::vector<double> actualSnapshotTimes;
    std::vector<StateRecord> records;
};

void writeCurveCsv(const FirstPassageExperimentConfig& config,
                   const std::vector<CurveSeries>& series) {
    std::ofstream output = openOutput(config.outputDirectory / "first_passage_curves.csv");
    output << "kernel_id,kernel_type,grid_step,grid_step_over_length_scale,bin_begin,bin_end,"
              "time,time_over_length_scale,at_risk,events,"
              "survival,survival_ci_lower,survival_ci_upper,first_passage_density,"
              "rice_w1,rice_w2_integral,rice_w2_absolute_error,rice_w2_converged,"
              "rice_density_order2,rice_survival_order1,rice_survival_order2,"
              "rice_hazard_order1,rice_hazard_order2,"
              "hazard_mc,hazard_ci_lower,hazard_ci_upper,cumulative_hazard_nelson_aalen,"
              "survival_from_hazard,start_conditioned_endpoint_survival,"
              "start_conditioned_sigma1_survival,pointwise_sigma2_survival,"
              "endpoint_conditioned_hazard_sigma1,pointwise_hazard_sigma2,"
              "rice_downcrossing_intensity\n";
    for (const CurveSeries& item : series) {
        const StationaryKernel kernel(item.kernel);
        const double rice = std::sqrt(kernel.derivativeVariance() /
                                      kernel.config().variance) / (2.0 * kPi);
        for (const CurvePoint& point : item.points) {
            output << csv(item.kernel.id) << ',' << item.kernel.type << ',' << item.step << ','
                   << item.step / item.kernel.lengthScale << ',' << point.begin << ','
                   << point.end << ',' << point.time << ','
                   << point.time / item.kernel.lengthScale << ','
                   << point.atRisk << ',' << point.events << ',' << point.survival << ','
                   << point.survivalLower << ',' << point.survivalUpper << ','
                   << point.density << ',' << point.riceW1 << ','
                   << point.riceW2Correction << ',' << point.riceW2AbsoluteError << ','
                   << (point.riceW2Converged ? 1 : 0) << ','
                   << point.riceDensityOrder2 << ',' << point.riceSurvivalOrder1 << ','
                   << point.riceSurvivalOrder2 << ',' << point.riceHazardOrder1 << ','
                   << point.riceHazardOrder2 << ',' << point.hazard << ','
                   << point.hazardLower << ','
                   << point.hazardUpper << ',' << point.cumulativeHazard << ','
                   << point.survivalFromHazard << ','
                   << point.startConditionedEndpointSurvival << ','
                   << point.startConditionedSigma1Survival << ','
                   << point.pointwiseSigma2Survival << ',' << point.endpointHazard << ','
                   << point.pointwiseHazard << ',' << rice << '\n';
        }
    }
}

void computeSeriesSummaries(const FirstPassageExperimentConfig& config,
                            std::vector<CurveSeries>& series) {
    for (CurveSeries& item : series) {
        double eventSum = 0.0;
        double restrictedSum = 0.0;
        for (const double time : item.eventTimes) {
            restrictedSum += std::min(time, config.maximumTime);
            if (std::isfinite(time)) {
                ++item.events;
                eventSum += time;
            }
        }
        item.restrictedMean = restrictedSum / item.eventTimes.size();
        if (item.events > 0) item.meanEventTime = eventSum / item.events;
        double endpointSquared = 0.0;
        double pointwiseSquared = 0.0;
        double truthSquared = 0.0;
        double endpointSurvivalSquared = 0.0;
        double sigma1SurvivalSquared = 0.0;
        double sigma2SurvivalSquared = 0.0;
        std::uint64_t survivalCount = 0;
        std::uint64_t count = 0;
        for (const CurvePoint& point : item.points) {
            if (point.atRisk < static_cast<std::uint64_t>(config.minimumRiskSetForError)) continue;
            if (!std::isfinite(point.hazard) || !std::isfinite(point.endpointHazard) ||
                !std::isfinite(point.pointwiseHazard)) continue;
            endpointSquared += (point.hazard - point.endpointHazard) *
                               (point.hazard - point.endpointHazard);
            pointwiseSquared += (point.hazard - point.pointwiseHazard) *
                                (point.hazard - point.pointwiseHazard);
            truthSquared += point.hazard * point.hazard;
            ++count;
        }
        for (const CurvePoint& point : item.points) {
            endpointSurvivalSquared +=
                (point.survival - point.startConditionedEndpointSurvival) *
                (point.survival - point.startConditionedEndpointSurvival);
            sigma1SurvivalSquared +=
                (point.survival - point.startConditionedSigma1Survival) *
                (point.survival - point.startConditionedSigma1Survival);
            sigma2SurvivalSquared +=
                (point.survival - point.pointwiseSigma2Survival) *
                (point.survival - point.pointwiseSigma2Survival);
            ++survivalCount;
        }
        if (count > 0) {
            item.endpointRmse = std::sqrt(endpointSquared / count);
            item.pointwiseRmse = std::sqrt(pointwiseSquared / count);
            if (truthSquared > 0.0) {
                const double truthRms = std::sqrt(truthSquared / count);
                item.endpointNrmse = item.endpointRmse / truthRms;
                item.pointwiseNrmse = item.pointwiseRmse / truthRms;
            }
        }
        if (survivalCount > 0) {
            item.endpointSurvivalRmse = std::sqrt(endpointSurvivalSquared / survivalCount);
            item.sigma1SurvivalRmse = std::sqrt(sigma1SurvivalSquared / survivalCount);
            item.sigma2SurvivalRmse = std::sqrt(sigma2SurvivalSquared / survivalCount);
        }
    }
}

void writeSummaryCsv(const FirstPassageExperimentConfig& config,
                     const std::vector<CurveSeries>& series) {
    std::ofstream output = openOutput(config.outputDirectory / "first_passage_summary.csv");
    output << "kernel_id,kernel_type,variance,length_scale,alpha,grid_step,"
              "grid_step_over_length_scale,trajectories,events,"
              "censored,event_probability,restricted_mean_survival_time,mean_observed_event_time,"
              "rice_downcrossing_intensity,pointwise_hazard_sigma2,endpoint_hazard_rmse,"
              "pointwise_hazard_rmse,endpoint_hazard_nrmse,pointwise_hazard_nrmse,"
              "endpoint_survival_rmse,sigma1_survival_rmse,sigma2_survival_rmse,"
              "embedding_size,minimum_embedding_eigenvalue\n";
    for (const CurveSeries& item : series) {
        const StationaryKernel kernel(item.kernel);
        const std::uint64_t censored = item.eventTimes.size() - item.events;
        const double rice = std::sqrt(kernel.derivativeVariance() /
                                      kernel.config().variance) / (2.0 * kPi);
        output << csv(item.kernel.id) << ',' << item.kernel.type << ','
               << item.kernel.variance << ',' << item.kernel.lengthScale << ','
               << item.kernel.alpha << ',' << item.step << ','
               << item.step / item.kernel.lengthScale << ',' << item.eventTimes.size() << ','
               << item.events << ',' << censored << ','
               << static_cast<double>(item.events) / item.eventTimes.size() << ','
               << item.restrictedMean << ',' << item.meanEventTime << ',' << rice << ','
               << pointwiseHazard(kernel, config.processMean, config.threshold) << ','
               << item.endpointRmse << ',' << item.pointwiseRmse << ','
               << item.endpointNrmse << ',' << item.pointwiseNrmse << ','
               << item.endpointSurvivalRmse << ',' << item.sigma1SurvivalRmse << ','
               << item.sigma2SurvivalRmse << ','
               << item.embeddingSize << ',' << item.minimumEigenvalue << '\n';
    }
}

void writeRawSamples(const FirstPassageExperimentConfig& config,
                     const std::vector<CurveSeries>& series) {
    if (!config.writeRawSamples) return;
    std::ofstream output = openOutput(config.outputDirectory / "first_passage_samples.csv");
    output << "kernel_id,kernel_type,grid_step,trajectory,event_time,censored\n";
    for (const CurveSeries& item : series) {
        for (std::size_t trajectory = 0; trajectory < item.eventTimes.size(); ++trajectory) {
            const double time = item.eventTimes[trajectory];
            output << csv(item.kernel.id) << ',' << item.kernel.type << ',' << item.step << ','
                   << trajectory << ',' << (std::isfinite(time) ? time : config.maximumTime) << ','
                   << (std::isfinite(time) ? 0 : 1) << '\n';
        }
    }
}

void writeStateOutputs(const FirstPassageExperimentConfig& config,
                       const std::vector<KernelStateRecords>& kernels) {
    if (!config.stateAnalysis.enabled) return;
    std::optional<std::ofstream> raw;
    if (config.stateAnalysis.writeSamples) {
        raw.emplace(openOutput(config.outputDirectory / "first_passage_state_samples.csv"));
        *raw << "kernel_id,kernel_type,trajectory,snapshot_time,value,derivative,"
                "event_in_future_window\n";
    }
    std::ofstream summary = openOutput(
        config.outputDirectory / "first_passage_state_summary.csv");
    summary << "kernel_id,kernel_type,snapshot_time,snapshot_time_over_length_scale,"
               "survivors,future_events,future_window,"
               "conditional_hazard,value_mean,value_stddev,derivative_mean,derivative_stddev,"
               "value_derivative_covariance,value_derivative_correlation\n";
    std::ofstream grid = openOutput(
        config.outputDirectory / "first_passage_state_hazard.csv");
    grid << "kernel_id,kernel_type,snapshot_time,value_bin,derivative_bin,value_min,value_max,"
            "derivative_min,derivative_max,at_risk,future_events,conditional_hazard\n";

    for (const KernelStateRecords& item : kernels) {
        const StationaryKernel kernel(item.kernel);
        const double valueMinimum = config.stateAnalysis.valueMinimum.value_or(config.threshold);
        const double valueMaximum = config.stateAnalysis.valueMaximum.value_or(
            std::max(valueMinimum + std::sqrt(item.kernel.variance),
                     config.processMean + 4.0 * std::sqrt(item.kernel.variance)));
        const double derivativeScale = std::sqrt(kernel.derivativeVariance());
        const double derivativeMinimum =
            config.stateAnalysis.derivativeMinimum.value_or(-4.0 * derivativeScale);
        const double derivativeMaximum =
            config.stateAnalysis.derivativeMaximum.value_or(4.0 * derivativeScale);
        const int valueBins = config.stateAnalysis.valueBins;
        const int derivativeBins = config.stateAnalysis.derivativeBins;

        for (std::size_t snapshot = 0; snapshot < item.actualSnapshotTimes.size(); ++snapshot) {
            std::vector<const StateRecord*> records;
            for (const StateRecord& record : item.records) {
                if (record.snapshot == snapshot) records.push_back(&record);
            }
            double valueSum = 0.0;
            double derivativeSum = 0.0;
            std::uint64_t futureEvents = 0;
            for (const StateRecord* record : records) {
                valueSum += record->value;
                derivativeSum += record->derivative;
                futureEvents += record->eventInWindow ? 1u : 0u;
                if (raw) {
                    *raw << csv(item.kernel.id) << ',' << item.kernel.type << ','
                         << record->trajectory << ',' << item.actualSnapshotTimes[snapshot] << ','
                         << record->value << ',' << record->derivative << ','
                         << (record->eventInWindow ? 1 : 0) << '\n';
                }
            }
            const double count = static_cast<double>(records.size());
            const double valueMean = count > 0.0 ? valueSum / count : 0.0;
            const double derivativeMean = count > 0.0 ? derivativeSum / count : 0.0;
            double valueVariance = 0.0;
            double derivativeVariance = 0.0;
            double covariance = 0.0;
            for (const StateRecord* record : records) {
                const double x = record->value - valueMean;
                const double v = record->derivative - derivativeMean;
                valueVariance += x * x;
                derivativeVariance += v * v;
                covariance += x * v;
            }
            if (records.size() > 1) {
                const double denominator = static_cast<double>(records.size() - 1);
                valueVariance /= denominator;
                derivativeVariance /= denominator;
                covariance /= denominator;
            }
            const double correlation = valueVariance > 0.0 && derivativeVariance > 0.0
                ? covariance / std::sqrt(valueVariance * derivativeVariance) : 0.0;
            summary << csv(item.kernel.id) << ',' << item.kernel.type << ','
                    << item.actualSnapshotTimes[snapshot] << ','
                    << item.actualSnapshotTimes[snapshot] / item.kernel.lengthScale << ','
                    << records.size() << ','
                    << futureEvents << ',' << config.stateAnalysis.futureWindow << ','
                    << (count > 0.0 ? futureEvents / (count * config.stateAnalysis.futureWindow) : 0.0)
                    << ',' << valueMean << ',' << std::sqrt(std::max(0.0, valueVariance)) << ','
                    << derivativeMean << ',' << std::sqrt(std::max(0.0, derivativeVariance))
                    << ',' << covariance << ',' << correlation << '\n';

            std::vector<std::uint64_t> risk(static_cast<std::size_t>(valueBins * derivativeBins));
            std::vector<std::uint64_t> events(risk.size());
            for (const StateRecord* record : records) {
                int xBin = static_cast<int>((record->value - valueMinimum) /
                    (valueMaximum - valueMinimum) * valueBins);
                int vBin = static_cast<int>((record->derivative - derivativeMinimum) /
                    (derivativeMaximum - derivativeMinimum) * derivativeBins);
                xBin = std::clamp(xBin, 0, valueBins - 1);
                vBin = std::clamp(vBin, 0, derivativeBins - 1);
                const std::size_t index = static_cast<std::size_t>(xBin * derivativeBins + vBin);
                ++risk[index];
                events[index] += record->eventInWindow ? 1u : 0u;
            }
            for (int xBin = 0; xBin < valueBins; ++xBin) {
                const double x0 = valueMinimum + (valueMaximum - valueMinimum) * xBin / valueBins;
                const double x1 = valueMinimum + (valueMaximum - valueMinimum) * (xBin + 1) / valueBins;
                for (int vBin = 0; vBin < derivativeBins; ++vBin) {
                    const double v0 = derivativeMinimum +
                        (derivativeMaximum - derivativeMinimum) * vBin / derivativeBins;
                    const double v1 = derivativeMinimum +
                        (derivativeMaximum - derivativeMinimum) * (vBin + 1) / derivativeBins;
                    const std::size_t index = static_cast<std::size_t>(xBin * derivativeBins + vBin);
                    const double hazard = risk[index] > 0
                        ? static_cast<double>(events[index]) /
                          (risk[index] * config.stateAnalysis.futureWindow) : 0.0;
                    grid << csv(item.kernel.id) << ',' << item.kernel.type << ','
                         << item.actualSnapshotTimes[snapshot] << ',' << xBin << ',' << vBin << ','
                         << x0 << ',' << x1 << ',' << v0 << ',' << v1 << ',' << risk[index] << ','
                         << events[index] << ',' << hazard << '\n';
                }
            }
        }
    }
}

void writeSvg(const FirstPassageExperimentConfig& config,
              const std::vector<CurveSeries>& series, bool hazardPlot) {
    const double finestStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    std::vector<const CurveSeries*> selected;
    for (const CurveSeries& item : series) {
        if (std::abs(item.step - finestStep) <= 1e-12 * finestStep) selected.push_back(&item);
    }
    double yMaximum = hazardPlot ? 0.0 : 1.0;
    if (hazardPlot) {
        for (const CurveSeries* item : selected) {
            for (const CurvePoint& point : item->points) {
                if (point.atRisk >= static_cast<std::uint64_t>(config.minimumRiskSetForError)) {
                    yMaximum = std::max({yMaximum, point.hazard, point.endpointHazard,
                                         point.pointwiseHazard});
                }
            }
        }
        if (!(yMaximum > 0.0) || !std::isfinite(yMaximum)) yMaximum = 1.0;
        yMaximum *= 1.08;
    }
    const std::vector<std::string> colors{
        "#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"};
    const std::filesystem::path path = config.outputDirectory /
        (hazardPlot ? "first_passage_hazard.svg" : "first_passage_survival.svg");
    std::ofstream output = openOutput(path);
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 470\">"
              "<rect width=\"800\" height=\"470\" fill=\"white\"/>"
              "<path d=\"M70 30 V400 H760\" fill=\"none\" stroke=\"#222\"/>"
              "<text x=\"370\" y=\"448\">t</text><text x=\"12\" y=\"24\">"
           << (hazardPlot ? "hazard" : "survival") << "</text>";
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const CurveSeries& item = *selected[index];
        const std::string& color = colors[index % colors.size()];
        output << "<polyline fill=\"none\" stroke=\"" << color
               << "\" stroke-width=\"2\" points=\"";
        for (const CurvePoint& point : item.points) {
            if (hazardPlot && point.atRisk <
                static_cast<std::uint64_t>(config.minimumRiskSetForError)) continue;
            const double value = hazardPlot ? point.hazard : point.survival;
            const double plotTime = hazardPlot ? point.time : point.end;
            const double x = 70.0 + 690.0 * plotTime / config.maximumTime;
            const double y = 400.0 - 360.0 * std::clamp(value / yMaximum, 0.0, 1.0);
            output << x << ',' << y << ' ';
        }
        output << "\"/><text x=\"520\" y=\"" << 52 + 20 * index
               << "\" fill=\"" << color << "\">" << xml(item.kernel.id)
               << "</text>";
        if (hazardPlot) {
            output << "<polyline fill=\"none\" stroke=\"" << color
                   << "\" stroke-dasharray=\"5 4\" opacity=\"0.75\" points=\"";
            for (const CurvePoint& point : item.points) {
                const double x = 70.0 + 690.0 * point.time / config.maximumTime;
                const double y = 400.0 - 360.0 *
                    std::clamp(point.endpointHazard / yMaximum, 0.0, 1.0);
                output << x << ',' << y << ' ';
            }
            output << "\"/>";
        } else {
            output << "<polyline fill=\"none\" stroke=\"" << color
                   << "\" stroke-dasharray=\"6 4\" opacity=\"0.85\" points=\"";
            for (const CurvePoint& point : item.points) {
                const double x = 70.0 + 690.0 * point.end / config.maximumTime;
                const double y = 400.0 - 360.0 *
                    std::clamp(point.startConditionedSigma1Survival, 0.0, 1.0);
                output << x << ',' << y << ' ';
            }
            output << "\"/><polyline fill=\"none\" stroke=\"" << color
                   << "\" stroke-dasharray=\"1 4\" opacity=\"0.7\" points=\"";
            for (const CurvePoint& point : item.points) {
                const double x = 70.0 + 690.0 * point.end / config.maximumTime;
                const double y = 400.0 - 360.0 *
                    std::clamp(point.startConditionedEndpointSurvival, 0.0, 1.0);
                output << x << ',' << y << ' ';
            }
            output << "\"/>";
        }
    }
    output << "<text x=\"70\" y=\"425\">solid: Monte Carlo";
    if (hazardPlot) output << "; dashed: endpoint-conditioned Sigma1";
    else output << "; dashed: survival from Sigma1; dotted: P(X_t &gt; b | X_0 = a)";
    output << "</text></svg>\n";
}

void writeRiceDensitySvg(const FirstPassageExperimentConfig& config,
                         const std::vector<CurveSeries>& series) {
    if (!config.riceSeries.enabled) return;
    const double finestStep = *std::min_element(
        config.stepSizes.begin(), config.stepSizes.end());
    std::vector<const CurveSeries*> selected;
    double yMinimum = 0.0;
    double yMaximum = 0.0;
    for (const CurveSeries& item : series) {
        if (std::abs(item.step - finestStep) > 1e-12 * finestStep) continue;
        selected.push_back(&item);
        for (const CurvePoint& point : item.points) {
            for (const double value : {point.density, point.riceW1,
                                       point.riceDensityOrder2}) {
                if (!std::isfinite(value)) continue;
                yMinimum = std::min(yMinimum, value);
                yMaximum = std::max(yMaximum, value);
            }
        }
    }
    if (!(yMaximum > yMinimum)) yMaximum = yMinimum + 1.0;
    const double padding = 0.06 * (yMaximum - yMinimum);
    yMinimum -= padding;
    yMaximum += padding;
    const auto plotY = [&](double value) {
        return 400.0 - 360.0 * (value - yMinimum) / (yMaximum - yMinimum);
    };
    const std::vector<std::string> colors{
        "#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"};
    std::ofstream output = openOutput(
        config.outputDirectory / "first_passage_rice_density.svg");
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 470\">"
              "<rect width=\"800\" height=\"470\" fill=\"white\"/>"
              "<path d=\"M70 30 V400 H760\" fill=\"none\" stroke=\"#222\"/>"
              "<text x=\"355\" y=\"448\">t</text>"
              "<text x=\"12\" y=\"24\">first-passage density</text>";
    if (yMinimum < 0.0 && yMaximum > 0.0) {
        output << "<path d=\"M70 " << plotY(0.0) << " H760\" stroke=\"#aaa\" "
                  "stroke-dasharray=\"2 3\"/>";
    }
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const CurveSeries& item = *selected[index];
        const std::string& color = colors[index % colors.size()];
        const auto polyline = [&](const char* dash, const auto& valueOf) {
            output << "<polyline fill=\"none\" stroke=\"" << color
                   << "\" stroke-width=\"2\"";
            if (dash[0] != '\0') output << " stroke-dasharray=\"" << dash << "\"";
            output << " points=\"";
            for (const CurvePoint& point : item.points) {
                const double value = valueOf(point);
                if (!std::isfinite(value)) continue;
                const double x = 70.0 + 690.0 * point.time / config.maximumTime;
                output << x << ',' << plotY(value) << ' ';
            }
            output << "\"/>";
        };
        polyline("", [](const CurvePoint& point) { return point.density; });
        polyline("6 4", [](const CurvePoint& point) { return point.riceW1; });
        if (config.riceSeries.maxOrder >= 2) {
            polyline("1 4", [](const CurvePoint& point) {
                return point.riceDensityOrder2;
            });
        }
        output << "<text x=\"535\" y=\"" << 52 + 20 * index
               << "\" fill=\"" << color << "\">" << xml(item.kernel.id)
               << "</text>";
    }
    output << "<text x=\"70\" y=\"425\">solid: Monte Carlo; dashed: Rice order 1; "
              "dotted: Rice order 2</text></svg>\n";
}

Json kernelJson(const FirstPassageKernelConfig& kernel) {
    Json result{{"id", kernel.id}, {"type", kernel.type},
                {"variance", kernel.variance}, {"length_scale", kernel.lengthScale}};
    if (kernel.type == "rational_quadratic") result["alpha"] = kernel.alpha;
    return result;
}

std::array<double, 2> collisionRange(const Json& value, const std::string& name) {
    if (!value.is_array() || value.size() != 2) {
        throw std::invalid_argument(name + " must contain [minimum, maximum]");
    }
    const std::array<double, 2> result{value[0].get<double>(), value[1].get<double>()};
    if (!std::isfinite(result[0]) || !std::isfinite(result[1]) ||
        !(result[1] > result[0])) {
        throw std::invalid_argument(name + " must be finite and increasing");
    }
    return result;
}

void shuffleIndices(std::vector<int>& values, Random& rng) {
    for (std::size_t i = values.size(); i > 1; --i) {
        const std::size_t selected = static_cast<std::size_t>(rng.integer() % i);
        std::swap(values[i - 1], values[selected]);
    }
}

std::vector<FirstPassageCollisionState> parseCollisionStates(
    const Json& initial, std::uint64_t seed) {
    if (!initial.contains("parameter_space") ||
        !initial.at("parameter_space").is_object()) {
        throw std::invalid_argument(
            "collision_state requires initial_condition.parameter_space");
    }
    const Json& space = initial.at("parameter_space");
    const std::string type = space.value("type", "explicit");
    std::vector<FirstPassageCollisionState> result;
    if (type == "explicit") {
        if (!space.contains("states") || !space.at("states").is_array() ||
            space.at("states").empty()) {
            throw std::invalid_argument("explicit collision parameter space requires states");
        }
        for (std::size_t index = 0; index < space.at("states").size(); ++index) {
            const Json& entry = space.at("states")[index];
            FirstPassageCollisionState state;
            state.id = entry.value("id", "state_" + std::to_string(index));
            state.beta0 = entry.at("beta_0").get<double>();
            state.betaMeanSlope = entry.at("beta_a").get<double>();
            state.betaCollisionSlope = entry.at("beta_g").get<double>();
            result.push_back(std::move(state));
        }
    } else if (type == "cartesian") {
        const std::vector<double> beta0 = space.at("beta_0").get<std::vector<double>>();
        const std::vector<double> betaA = space.at("beta_a").get<std::vector<double>>();
        const std::vector<double> betaG = space.at("beta_g").get<std::vector<double>>();
        if (beta0.empty() || betaA.empty() || betaG.empty()) {
            throw std::invalid_argument("cartesian collision parameter axes must be nonempty");
        }
        for (double b0 : beta0) for (double ba : betaA) for (double bg : betaG) {
            FirstPassageCollisionState state;
            state.id = "state_" + std::to_string(result.size());
            state.beta0 = b0;
            state.betaMeanSlope = ba;
            state.betaCollisionSlope = bg;
            result.push_back(std::move(state));
        }
    } else if (type == "latin_hypercube") {
        const int count = space.at("count").get<int>();
        if (count < 1) throw std::invalid_argument("latin_hypercube count must be positive");
        const std::array<double, 2> beta0 = collisionRange(
            space.at("beta_0_range"), "parameter_space.beta_0_range");
        const std::array<double, 2> betaA = collisionRange(
            space.at("beta_a_range"), "parameter_space.beta_a_range");
        const std::array<double, 2> betaG = collisionRange(
            space.at("beta_g_range"), "parameter_space.beta_g_range");
        Random rng(mixSeed(seed ^ 0x6c6174696e687970ULL));
        std::array<std::vector<int>, 3> permutations;
        for (auto& permutation : permutations) {
            permutation.resize(static_cast<std::size_t>(count));
            std::iota(permutation.begin(), permutation.end(), 0);
            shuffleIndices(permutation, rng);
        }
        const auto sample = [&](const std::array<double, 2>& range, int stratum) {
            const double coordinate =
                (static_cast<double>(stratum) + rng.uniform01()) / count;
            return range[0] + coordinate * (range[1] - range[0]);
        };
        for (int index = 0; index < count; ++index) {
            FirstPassageCollisionState state;
            state.id = "lhs_" + std::to_string(index);
            state.beta0 = sample(beta0, permutations[0][static_cast<std::size_t>(index)]);
            state.betaMeanSlope = sample(betaA, permutations[1][static_cast<std::size_t>(index)]);
            state.betaCollisionSlope = sample(
                betaG, permutations[2][static_cast<std::size_t>(index)]);
            result.push_back(std::move(state));
        }
    } else {
        throw std::invalid_argument(
            "collision parameter_space.type must be explicit, cartesian, or latin_hypercube");
    }
    return result;
}

} // namespace

FirstPassageExperimentConfig loadFirstPassageExperimentConfig(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open first-passage config: " + path.string());
    Json root;
    stream >> root;
    FirstPassageExperimentConfig config;
    config.schemaVersion = root.value("schema_version", config.schemaVersion);
    if (config.schemaVersion != 1) throw std::invalid_argument("unsupported config schema version");
    config.seed = root.value("seed", config.seed);
    config.outputDirectory = root.value("output_directory", config.outputDirectory.string());
    if (!root.contains("first_passage") || !root.at("first_passage").is_object()) {
        throw std::invalid_argument("first-passage command requires a first_passage object");
    }
    const Json& experiment = root.at("first_passage");
    if (experiment.contains("process")) {
        const Json& process = experiment.at("process");
        config.processMean = process.value("mean", config.processMean);
        config.threshold = process.value("threshold", config.threshold);
    }
    if (experiment.contains("initial_condition")) {
        const Json& initial = experiment.at("initial_condition");
        const std::string type = initial.value("type", "fixed_value");
        config.initialConditionType = type;
        if (type == "fixed_value") {
            config.initialValue = initial.at("value").get<double>();
        } else if (type == "collision_state") {
            config.collisionStates = parseCollisionStates(initial, config.seed);
            // These legacy diagnostics have different conditioning semantics.
            config.riceSeries.enabled = false;
            config.stateAnalysis.enabled = false;
            config.writeRawSamples = true;
        } else {
            throw std::invalid_argument(
                "first_passage.initial_condition.type must be fixed_value or collision_state");
        }
    }
    if (experiment.contains("grid")) {
        const Json& grid = experiment.at("grid");
        config.maximumTime = grid.value("max_time", config.maximumTime);
        if (grid.contains("step_sizes")) {
            config.stepSizes = grid.at("step_sizes").get<std::vector<double>>();
        }
    }
    if (experiment.contains("curve")) {
        config.curveBins = experiment.at("curve").value("bins", config.curveBins);
    }
    if (experiment.contains("visualization")) {
        const Json& visualization = experiment.at("visualization");
        if (!visualization.is_object()) {
            throw std::invalid_argument("first_passage.visualization must be an object");
        }
        config.collisionVisualization.comparison = visualization.value(
            "comparison", config.collisionVisualization.comparison);
        if (visualization.contains("kernel_id")) {
            config.collisionVisualization.kernelId =
                visualization.at("kernel_id").get<std::string>();
        }
        if (visualization.contains("state_ids")) {
            config.collisionVisualization.stateIds =
                visualization.at("state_ids").get<std::vector<std::string>>();
        }
    }
    if (experiment.contains("monte_carlo")) {
        const Json& monteCarlo = experiment.at("monte_carlo");
        config.trajectories = monteCarlo.value("trajectories", config.trajectories);
        config.threadCount = monteCarlo.value("thread_count", config.threadCount);
        config.confidenceLevel = monteCarlo.value("confidence_level", config.confidenceLevel);
        config.writeRawSamples = monteCarlo.value("write_raw_samples", config.writeRawSamples);
        config.minimumRiskSetForError = monteCarlo.value(
            "minimum_risk_set_for_error", config.minimumRiskSetForError);
    }
    if (experiment.contains("sampler")) {
        const Json& sampler = experiment.at("sampler");
        const std::string defaultType = config.initialConditionType == "collision_state"
            ? "collision_state_auto" : "exact_grid_circulant";
        const std::string type = sampler.value("type", defaultType);
        if (config.initialConditionType == "fixed_value" && type != "exact_grid_circulant") {
            throw std::invalid_argument(
                "fixed_value first passage requires sampler.type=exact_grid_circulant");
        }
        if (config.initialConditionType == "collision_state" &&
            type != "collision_state_auto") {
            throw std::invalid_argument(
                "collision_state requires sampler.type=collision_state_auto");
        }
        config.maximumEmbeddingExpansions = sampler.value(
            "max_embedding_expansions", config.maximumEmbeddingExpansions);
        config.collisionSampler.minimumStep = sampler.value(
            "minimum_step", config.collisionSampler.minimumStep);
        config.collisionSampler.crossingTolerance = sampler.value(
            "crossing_tolerance", config.collisionSampler.crossingTolerance);
        config.collisionSampler.bridgeSigmaMargin = sampler.value(
            "bridge_sigma_margin", config.collisionSampler.bridgeSigmaMargin);
        config.collisionSampler.maximumRefinementDepth = sampler.value(
            "max_refinement_depth", config.collisionSampler.maximumRefinementDepth);
    }
    if (experiment.contains("rice_series")) {
        const Json& rice = experiment.at("rice_series");
        config.riceSeries.enabled = rice.value("enabled", config.riceSeries.enabled);
        config.riceSeries.maxOrder = rice.value("max_order", config.riceSeries.maxOrder);
        config.riceSeries.relativeTolerance = rice.value(
            "relative_tolerance", config.riceSeries.relativeTolerance);
        config.riceSeries.absoluteTolerance = rice.value(
            "absolute_tolerance", config.riceSeries.absoluteTolerance);
        config.riceSeries.maximumQuadratureSubdivisions = rice.value(
            "max_quadrature_subdivisions",
            config.riceSeries.maximumQuadratureSubdivisions);
        config.riceSeries.covarianceRoundoffMultiplier = rice.value(
            "covariance_roundoff_multiplier",
            config.riceSeries.covarianceRoundoffMultiplier);
    }
    if (experiment.contains("state_analysis")) {
        const Json& state = experiment.at("state_analysis");
        config.stateAnalysis.enabled = state.value("enabled", config.stateAnalysis.enabled);
        if (state.contains("snapshot_times")) {
            config.stateAnalysis.snapshotTimes =
                state.at("snapshot_times").get<std::vector<double>>();
        }
        config.stateAnalysis.futureWindow = state.value(
            "future_window", config.stateAnalysis.futureWindow);
        config.stateAnalysis.valueBins = state.value(
            "value_bins", config.stateAnalysis.valueBins);
        config.stateAnalysis.derivativeBins = state.value(
            "derivative_bins", config.stateAnalysis.derivativeBins);
        config.stateAnalysis.writeSamples = state.value(
            "write_samples", config.stateAnalysis.writeSamples);
        if (state.contains("value_range")) {
            const std::vector<double> range = state.at("value_range").get<std::vector<double>>();
            if (range.size() != 2) throw std::invalid_argument("state_analysis.value_range needs two values");
            config.stateAnalysis.valueMinimum = range[0];
            config.stateAnalysis.valueMaximum = range[1];
        }
        if (state.contains("derivative_range")) {
            const std::vector<double> range = state.at("derivative_range").get<std::vector<double>>();
            if (range.size() != 2) throw std::invalid_argument("state_analysis.derivative_range needs two values");
            config.stateAnalysis.derivativeMinimum = range[0];
            config.stateAnalysis.derivativeMaximum = range[1];
        }
    }
    if (!experiment.contains("kernels") || !experiment.at("kernels").is_array() ||
        experiment.at("kernels").empty()) {
        throw std::invalid_argument("first_passage.kernels must be a nonempty array");
    }
    std::unordered_set<std::string> ids;
    for (const Json& source : experiment.at("kernels")) {
        FirstPassageKernelConfig kernel;
        kernel.id = source.at("id").get<std::string>();
        kernel.type = source.at("type").get<std::string>();
        kernel.variance = source.value("variance", kernel.variance);
        kernel.lengthScale = source.at("length_scale").get<double>();
        kernel.alpha = source.value("alpha", kernel.alpha);
        if (kernel.id.empty() || !ids.insert(kernel.id).second) {
            throw std::invalid_argument("first_passage kernel ids must be nonempty and unique");
        }
        if (kernel.type != "squared_exponential" && kernel.type != "matern_3_2" &&
            kernel.type != "matern_5_2" && kernel.type != "rational_quadratic") {
            throw std::invalid_argument("unknown first-passage kernel type: " + kernel.type);
        }
        requirePositive(kernel.variance, "kernel.variance");
        requirePositive(kernel.lengthScale, "kernel.length_scale");
        if (kernel.type == "rational_quadratic") requirePositive(kernel.alpha, "kernel.alpha");
        config.kernels.push_back(std::move(kernel));
    }

    requireFinite(config.processMean, "first_passage.process.mean");
    requireFinite(config.threshold, "first_passage.process.threshold");
    if (config.initialConditionType == "fixed_value") {
        requireFinite(config.initialValue, "first_passage.initial_condition.value");
        if (!(config.initialValue > config.threshold)) {
            throw std::invalid_argument("initial_condition.value must be above process.threshold");
        }
    } else {
        // This mode exists to produce per-realization censored training data;
        // Aggregated risk-set counts are sufficient for interval-likelihood
        // training; raw realizations remain available as an optional diagnostic.
        if (config.collisionStates.empty()) {
            throw std::invalid_argument("collision-state parameter space cannot be empty");
        }
        if (config.threshold != 0.0 || config.processMean != 0.0) {
            throw std::invalid_argument(
                "collision_state uses dimensionless affine means and requires process.mean=threshold=0");
        }
        std::unordered_set<std::string> stateIds;
        for (const FirstPassageCollisionState& state : config.collisionStates) {
            if (state.id.empty() || !stateIds.insert(state.id).second) {
                throw std::invalid_argument("collision state ids must be nonempty and unique");
            }
            requireFinite(state.beta0, "collision beta_0");
            requireFinite(state.betaMeanSlope, "collision beta_a");
            requirePositive(state.betaCollisionSlope, "collision beta_g");
        }
        requirePositive(config.collisionSampler.minimumStep, "sampler.minimum_step");
        requirePositive(config.collisionSampler.crossingTolerance,
                        "sampler.crossing_tolerance");
        requirePositive(config.collisionSampler.bridgeSigmaMargin,
                        "sampler.bridge_sigma_margin");
        if (config.collisionSampler.maximumRefinementDepth < 1) {
            throw std::invalid_argument("sampler.max_refinement_depth must be positive");
        }
        if (config.riceSeries.enabled || config.stateAnalysis.enabled) {
            throw std::invalid_argument(
                "Rice and survivor-state diagnostics must be disabled for collision_state");
        }
        const std::string& comparison = config.collisionVisualization.comparison;
        if (comparison != "kernels" && comparison != "states") {
            throw std::invalid_argument(
                "visualization.comparison must be kernels or states");
        }
        std::unordered_set<std::string> selectedStateIds;
        for (const std::string& stateId : config.collisionVisualization.stateIds) {
            if (!selectedStateIds.insert(stateId).second) {
                throw std::invalid_argument("visualization.state_ids must be unique");
            }
            if (stateIds.find(stateId) == stateIds.end()) {
                throw std::invalid_argument(
                    "visualization references unknown state_id: " + stateId);
            }
        }
        if (comparison == "states") {
            if (!config.collisionVisualization.kernelId ||
                config.collisionVisualization.kernelId->empty()) {
                throw std::invalid_argument(
                    "state comparison requires visualization.kernel_id");
            }
            if (ids.find(*config.collisionVisualization.kernelId) == ids.end()) {
                throw std::invalid_argument(
                    "visualization references unknown kernel_id: " +
                    *config.collisionVisualization.kernelId);
            }
            if (config.collisionVisualization.stateIds.empty()) {
                throw std::invalid_argument(
                    "state comparison requires nonempty visualization.state_ids");
            }
        } else {
            if (config.collisionVisualization.kernelId) {
                throw std::invalid_argument(
                    "kernel comparison does not accept visualization.kernel_id");
            }
            if (config.collisionVisualization.stateIds.size() > 1) {
                throw std::invalid_argument(
                    "kernel comparison accepts at most one visualization.state_id");
            }
        }
    }
    requirePositive(config.maximumTime, "first_passage.grid.max_time");
    if (config.stepSizes.empty()) throw std::invalid_argument("grid.step_sizes cannot be empty");
    for (double step : config.stepSizes) requirePositive(step, "first_passage grid step");
    std::sort(config.stepSizes.begin(), config.stepSizes.end(), std::greater<double>());
    config.stepSizes.erase(std::unique(config.stepSizes.begin(), config.stepSizes.end(),
        [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(a, b); }),
        config.stepSizes.end());
    const double fineStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    if (config.initialConditionType == "fixed_value" &&
        !closeInteger(config.maximumTime / fineStep)) {
        throw std::invalid_argument("grid.max_time must be an integer multiple of the finest step");
    }
    for (double step : config.stepSizes) {
        if (config.initialConditionType == "fixed_value" &&
            (!closeInteger(step / fineStep) || !closeInteger(config.maximumTime / step))) {
            throw std::invalid_argument(
                "every grid step must be an integer multiple of the finest step and divide max_time");
        }
    }
    if (config.initialConditionType == "collision_state") {
        const double largestStep = *std::max_element(config.stepSizes.begin(), config.stepSizes.end());
        for (double step : config.stepSizes) {
            if (!closeInteger(config.maximumTime / step)) {
                throw std::invalid_argument(
                    "collision_state grid steps must divide grid.max_time");
            }
        }
        const bool hasMatern32 = std::any_of(
            config.kernels.begin(), config.kernels.end(),
            [](const FirstPassageKernelConfig& kernel) {
                return kernel.type == "matern_3_2";
            });
        if (hasMatern32 &&
            (config.collisionSampler.minimumStep > fineStep ||
             config.collisionSampler.crossingTolerance >= config.collisionSampler.minimumStep ||
             std::ldexp(largestStep, -config.collisionSampler.maximumRefinementDepth) >
                 config.collisionSampler.minimumStep)) {
            throw std::invalid_argument(
                "collision sampler requires crossing_tolerance < minimum_step <= finest grid step "
                "and enough refinement depth for the largest step");
        }
    }
    if (config.curveBins < 2 || config.trajectories < 1 || config.threadCount < 0 ||
        config.maximumEmbeddingExpansions < 0 || config.minimumRiskSetForError < 1 ||
        !(config.confidenceLevel > 0.0 && config.confidenceLevel < 1.0)) {
        throw std::invalid_argument("invalid first-passage experiment budget");
    }
    if (config.riceSeries.maxOrder < 1 || config.riceSeries.maxOrder > 2 ||
        !(config.riceSeries.relativeTolerance > 0.0) ||
        !(config.riceSeries.absoluteTolerance > 0.0) ||
        config.riceSeries.maximumQuadratureSubdivisions < 1 ||
        !(config.riceSeries.covarianceRoundoffMultiplier >= 1.0)) {
        throw std::invalid_argument(
            "rice_series currently requires max_order 1 or 2 and positive tolerances/budget");
    }
    if (config.stateAnalysis.enabled) {
        requirePositive(config.stateAnalysis.futureWindow, "state_analysis.future_window");
        if (config.stateAnalysis.valueBins < 1 || config.stateAnalysis.derivativeBins < 1 ||
            config.stateAnalysis.snapshotTimes.empty()) {
            throw std::invalid_argument("invalid state_analysis grid");
        }
        for (const double time : config.stateAnalysis.snapshotTimes) {
            if (!(time >= fineStep && time + config.stateAnalysis.futureWindow <= config.maximumTime - fineStep)) {
                throw std::invalid_argument(
                    "state snapshot must leave one fine-grid point on each side and its future window inside max_time");
            }
        }
        if (config.stateAnalysis.valueMinimum && config.stateAnalysis.valueMaximum &&
            !(*config.stateAnalysis.valueMaximum > *config.stateAnalysis.valueMinimum)) {
            throw std::invalid_argument("state_analysis.value_range must be increasing");
        }
        if (config.stateAnalysis.derivativeMinimum && config.stateAnalysis.derivativeMaximum &&
            !(*config.stateAnalysis.derivativeMaximum > *config.stateAnalysis.derivativeMinimum)) {
            throw std::invalid_argument("state_analysis.derivative_range must be increasing");
        }
    }
    return config;
}

void writeResolvedFirstPassageConfig(const FirstPassageExperimentConfig& config,
                                     const std::filesystem::path& path) {
    Json kernels = Json::array();
    for (const FirstPassageKernelConfig& kernel : config.kernels) {
        kernels.push_back(kernelJson(kernel));
    }
    Json state{
        {"enabled", config.stateAnalysis.enabled},
        {"snapshot_times", config.stateAnalysis.snapshotTimes},
        {"future_window", config.stateAnalysis.futureWindow},
        {"value_bins", config.stateAnalysis.valueBins},
        {"derivative_bins", config.stateAnalysis.derivativeBins},
        {"write_samples", config.stateAnalysis.writeSamples}};
    if (config.stateAnalysis.valueMinimum && config.stateAnalysis.valueMaximum) {
        state["value_range"] = {*config.stateAnalysis.valueMinimum,
                                *config.stateAnalysis.valueMaximum};
    }
    if (config.stateAnalysis.derivativeMinimum && config.stateAnalysis.derivativeMaximum) {
        state["derivative_range"] = {*config.stateAnalysis.derivativeMinimum,
                                     *config.stateAnalysis.derivativeMaximum};
    }
    Json initial;
    Json sampler;
    Json visualization{{"comparison", config.collisionVisualization.comparison},
                       {"state_ids", config.collisionVisualization.stateIds}};
    if (config.collisionVisualization.kernelId) {
        visualization["kernel_id"] = *config.collisionVisualization.kernelId;
    }
    if (config.initialConditionType == "collision_state") {
        Json states = Json::array();
        for (const FirstPassageCollisionState& item : config.collisionStates) {
            states.push_back({{"id", item.id}, {"beta_0", item.beta0},
                              {"beta_a", item.betaMeanSlope},
                              {"beta_g", item.betaCollisionSlope}});
        }
        initial = {{"type", "collision_state"},
                   {"parameter_space", {{"type", "explicit"},
                                        {"states", std::move(states)}}}};
        sampler = {{"type", "collision_state_auto"},
                   {"max_embedding_expansions", config.maximumEmbeddingExpansions},
                   {"minimum_step", config.collisionSampler.minimumStep},
                   {"crossing_tolerance", config.collisionSampler.crossingTolerance},
                   {"bridge_sigma_margin", config.collisionSampler.bridgeSigmaMargin},
                   {"max_refinement_depth",
                    config.collisionSampler.maximumRefinementDepth}};
    } else {
        initial = {{"type", "fixed_value"}, {"value", config.initialValue}};
        sampler = {{"type", "exact_grid_circulant"},
                   {"max_embedding_expansions", config.maximumEmbeddingExpansions}};
    }
    Json root{
        {"schema_version", config.schemaVersion},
        {"seed", config.seed},
        {"first_passage", {
            {"process", {{"mean", config.processMean}, {"threshold", config.threshold}}},
            {"initial_condition", std::move(initial)},
            {"grid", {{"max_time", config.maximumTime}, {"step_sizes", config.stepSizes}}},
            {"curve", {{"bins", config.curveBins}}},
            {"visualization", std::move(visualization)},
            {"monte_carlo", {
                {"trajectories", config.trajectories},
                {"thread_count", config.threadCount},
                {"confidence_level", config.confidenceLevel},
                {"write_raw_samples", config.writeRawSamples},
                {"minimum_risk_set_for_error", config.minimumRiskSetForError}}},
            {"sampler", std::move(sampler)},
            {"rice_series", {
                {"enabled", config.riceSeries.enabled},
                {"max_order", config.riceSeries.maxOrder},
                {"relative_tolerance", config.riceSeries.relativeTolerance},
                {"absolute_tolerance", config.riceSeries.absoluteTolerance},
                {"max_quadrature_subdivisions",
                 config.riceSeries.maximumQuadratureSubdivisions},
                {"covariance_roundoff_multiplier",
                 config.riceSeries.covarianceRoundoffMultiplier}}},
            {"state_analysis", std::move(state)},
            {"kernels", std::move(kernels)}}},
        {"output_directory", config.outputDirectory.string()}};
    std::ofstream output = openOutput(path);
    output << std::setw(2) << root << '\n';
}

namespace {

const char* collisionSamplerName(const FirstPassageKernelConfig& kernel) {
    return kernel.type == "matern_3_2"
        ? "matern32_state_space" : "conditioned_grid_circulant";
}

const char* collisionSlopeMethod(const FirstPassageKernelConfig& kernel) {
    return kernel.type == "matern_3_2" ? "state_bridge_hermite" : "grid_cubic_hermite";
}

std::vector<CollisionSample> generateCollisionSamples(
    const FirstPassageExperimentConfig& config,
    const FirstPassageKernelConfig& kernel,
    const FirstPassageCollisionState& state,
    std::size_t kernelIndex, std::size_t stateIndex,
    std::size_t resolutionIndex, double step) {
    std::vector<CollisionSample> result(static_cast<std::size_t>(config.trajectories));
    unsigned int workerCount = config.threadCount > 0
        ? static_cast<unsigned int>(config.threadCount) : std::thread::hardware_concurrency();
    workerCount = std::max(1u, std::min(workerCount,
        static_cast<unsigned int>(config.trajectories)));
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(workerCount);
    workers.reserve(workerCount);
    FirstPassageKernelConfig normalizedKernel = kernel;
    normalizedKernel.variance = 1.0;
    normalizedKernel.lengthScale = 1.0;
    const StationaryKernel dimensionlessKernel(normalizedKernel);
    const std::size_t gridSamples = static_cast<std::size_t>(
        std::llround(config.maximumTime / step)) + 1;
    for (unsigned int worker = 0; worker < workerCount; ++worker) {
        const int begin = static_cast<int>(
            static_cast<std::uint64_t>(config.trajectories) * worker / workerCount);
        const int end = static_cast<int>(
            static_cast<std::uint64_t>(config.trajectories) * (worker + 1) / workerCount);
        workers.emplace_back([&, worker, begin, end]() {
            try {
                std::optional<CirculantSampler> gridSampler;
                if (kernel.type != "matern_3_2") {
                    gridSampler.emplace(dimensionlessKernel, gridSamples, step,
                                        config.maximumEmbeddingExpansions);
                }
                for (int trajectory = begin; trajectory < end; ++trajectory) {
                    const std::uint64_t seed = collisionSeed(
                        config.seed, kernelIndex, stateIndex, resolutionIndex,
                        static_cast<std::uint64_t>(trajectory));
                    result[static_cast<std::size_t>(trajectory)] =
                        kernel.type == "matern_3_2"
                        ? sampleCollisionFirstPassage(
                            state, config.collisionSampler, config.maximumTime, step, seed)
                        : sampleGridCollisionFirstPassage(
                            *gridSampler, state, config.maximumTime, step, seed);
                }
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    for (const std::exception_ptr& error : errors) if (error) std::rethrow_exception(error);
    return result;
}

double sampleQuantile(std::vector<double> values, double probability) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double coordinate = probability * static_cast<double>(values.size() - 1);
    const std::size_t left = static_cast<std::size_t>(std::floor(coordinate));
    const std::size_t right = std::min(values.size() - 1, left + 1);
    const double fraction = coordinate - static_cast<double>(left);
    return values[left] + fraction * (values[right] - values[left]);
}

void writeCollisionSamples(std::ofstream& output,
                           const FirstPassageExperimentConfig& config,
                           const FirstPassageKernelConfig& kernel,
                           const FirstPassageCollisionState& state,
                           double step, bool trainingResolution,
                           const std::vector<CollisionSample>& samples) {
    const double sigma = std::sqrt(kernel.variance);
    const bool stateSpace = kernel.type == "matern_3_2";
    const double effectiveTolerance = stateSpace
        ? config.collisionSampler.crossingTolerance : step;
    const double effectiveMinimumStep = stateSpace
        ? config.collisionSampler.minimumStep : step;
    const double effectiveBridgeMargin = stateSpace
        ? config.collisionSampler.bridgeSigmaMargin : 0.0;
    for (std::size_t trajectory = 0; trajectory < samples.size(); ++trajectory) {
        const CollisionSample& sample = samples[trajectory];
        output << csv(kernel.id) << ',' << kernel.type << ','
               << collisionSamplerName(kernel) << ',' << collisionSlopeMethod(kernel) << ','
               << sigma << ',' << kernel.lengthScale << ',' << kernel.alpha << ','
               << csv(state.id) << ','
               << state.beta0 << ',' << state.betaMeanSlope << ','
               << state.betaCollisionSlope << ',' << config.maximumTime << ','
               << step << ',' << (trainingResolution ? 1 : 0) << ',' << trajectory << ','
               << sample.eventQ << ',' << sample.eventQ * kernel.lengthScale << ','
               << (sample.event ? 1 : 0) << ',' << (sample.event ? 0 : 1) << ',';
        if (sample.event) {
            output << sample.crossingSlope << ','
                   << -sample.crossingSlope * sigma / kernel.lengthScale;
        } else {
            output << ',';
        }
        output << ',' << sample.seed << ',' << effectiveTolerance
               << ',' << effectiveMinimumStep << ',' << effectiveBridgeMargin << ','
               << sample.transitions << ',' << sample.bridgeRefinements << ','
               << sample.deepestRefinement << '\n';
    }
}

void writeCollisionSummary(std::ofstream& output,
                           const FirstPassageExperimentConfig& config,
                           const FirstPassageKernelConfig& kernel,
                           const FirstPassageCollisionState& state,
                           double step, bool trainingResolution,
                           const std::vector<CollisionSample>& samples) {
    std::vector<double> eventTimes;
    std::vector<double> slopes;
    double restrictedMean = 0.0;
    double refinements = 0.0;
    double transitions = 0.0;
    for (const CollisionSample& sample : samples) {
        restrictedMean += sample.eventQ;
        refinements += static_cast<double>(sample.bridgeRefinements);
        transitions += static_cast<double>(sample.transitions);
        if (sample.event) {
            eventTimes.push_back(sample.eventQ);
            slopes.push_back(sample.crossingSlope);
        }
    }
    const double count = static_cast<double>(samples.size());
    output << csv(kernel.id) << ',' << kernel.type << ','
           << collisionSamplerName(kernel) << ',' << collisionSlopeMethod(kernel) << ','
           << std::sqrt(kernel.variance) << ',' << kernel.lengthScale << ','
           << kernel.alpha << ','
           << csv(state.id) << ',' << state.beta0
           << ',' << state.betaMeanSlope << ',' << state.betaCollisionSlope << ','
           << config.maximumTime << ',' << step << ',' << (trainingResolution ? 1 : 0)
           << ',' << samples.size() << ','
           << eventTimes.size() << ',' << samples.size() - eventTimes.size() << ','
           << eventTimes.size() / count << ',' << restrictedMean / count << ','
           << sampleQuantile(eventTimes, 0.1) << ',' << sampleQuantile(eventTimes, 0.5)
           << ',' << sampleQuantile(eventTimes, 0.9) << ','
           << sampleQuantile(slopes, 0.1) << ',' << sampleQuantile(slopes, 0.5) << ','
           << sampleQuantile(slopes, 0.9) << ',' << transitions / count << ','
           << refinements / count << '\n';
}

struct CollisionCurvePoint {
    double q = 0.0;
    std::uint64_t atRisk = 0;
    double survival = 0.0;
    double density = 0.0;
    double hazard = 0.0;
    double cumulativeHazard = 0.0;
};

struct CollisionCurveSeries {
    FirstPassageKernelConfig kernel;
    std::string stateId;
    std::string legendLabel;
    std::vector<CollisionCurvePoint> points;
};

std::vector<CollisionCurvePoint> writeCollisionCurves(std::ofstream& output,
                          const FirstPassageExperimentConfig& config,
                          const FirstPassageKernelConfig& kernel,
                          const FirstPassageCollisionState& state,
                          double step, bool trainingResolution,
                          const std::vector<CollisionSample>& samples) {
    std::vector<CollisionCurvePoint> points;
    points.reserve(static_cast<std::size_t>(config.curveBins));
    double nelsonAalen = 0.0;
    double productLimit = 1.0;
    const double width = config.maximumTime / config.curveBins;
    for (int bin = 0; bin < config.curveBins; ++bin) {
        const double begin = width * bin;
        const double end = width * (bin + 1);
        std::uint64_t atRisk = 0;
        std::uint64_t events = 0;
        std::uint64_t survivors = 0;
        for (const CollisionSample& sample : samples) {
            if (!sample.event || sample.eventQ > begin) ++atRisk;
            if (sample.event && sample.eventQ > begin && sample.eventQ <= end) ++events;
            if (!sample.event || sample.eventQ > end) ++survivors;
        }
        const double eventProbability = atRisk > 0
            ? static_cast<double>(events) / atRisk : 0.0;
        const double hazard = eventProbability / width;
        nelsonAalen += eventProbability;
        productLimit *= std::max(0.0, 1.0 - eventProbability);
        const double survival = static_cast<double>(survivors) / samples.size();
        const double cumulativeHazard = productLimit > 0.0
            ? -std::log(productLimit) : std::numeric_limits<double>::infinity();
        const double density = static_cast<double>(events) / (samples.size() * width);
        points.push_back({end, atRisk, survival, density, hazard, cumulativeHazard});
        output << csv(kernel.id) << ',' << kernel.type << ','
               << collisionSamplerName(kernel) << ',' << collisionSlopeMethod(kernel) << ','
               << std::sqrt(kernel.variance) << ',' << kernel.lengthScale << ','
               << kernel.alpha << ','
               << csv(state.id) << ','
               << state.beta0 << ',' << state.betaMeanSlope << ','
               << state.betaCollisionSlope << ',' << step << ','
               << (trainingResolution ? 1 : 0) << ',' << bin << ',' << begin << ','
               << end << ',' << atRisk << ',' << events << ',' << survival << ','
               << density << ','
               << hazard << ',' << nelsonAalen << ',' << cumulativeHazard << '\n';
    }
    return points;
}

enum class CollisionPlotField { Survival, Hazard, CumulativeHazard };

void writeCollisionSvg(const FirstPassageExperimentConfig& config,
                       const std::vector<CollisionCurveSeries>& series,
                       CollisionPlotField field) {
    if (series.empty()) return;
    const char* fieldName = field == CollisionPlotField::Survival ? "survival" :
        (field == CollisionPlotField::Hazard ? "hazard" : "cumulative hazard");
    const char* fileName = field == CollisionPlotField::Survival
        ? "first_passage_survival.svg" :
        (field == CollisionPlotField::Hazard ? "first_passage_hazard.svg" :
         "first_passage_cumulative_hazard.svg");
    double yMaximum = field == CollisionPlotField::Survival ? 1.0 : 0.0;
    for (const CollisionCurveSeries& item : series) {
        for (const CollisionCurvePoint& point : item.points) {
            if (field == CollisionPlotField::Hazard &&
                point.atRisk < static_cast<std::uint64_t>(config.minimumRiskSetForError)) {
                continue;
            }
            const double value = field == CollisionPlotField::Survival ? point.survival :
                (field == CollisionPlotField::Hazard ? point.hazard : point.cumulativeHazard);
            if (std::isfinite(value)) yMaximum = std::max(yMaximum, value);
        }
    }
    if (!(yMaximum > 0.0)) yMaximum = 1.0;
    if (field != CollisionPlotField::Survival) yMaximum *= 1.08;
    const std::vector<std::string> colors{
        "#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f",
        "#ad1457", "#558b2f", "#4527a0", "#6d4c41"};
    std::ofstream output = openOutput(config.outputDirectory / fileName);
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 470\">"
              "<rect width=\"800\" height=\"470\" fill=\"white\"/>"
              "<path d=\"M70 30 V400 H760\" fill=\"none\" stroke=\"#222\"/>"
              "<text x=\"365\" y=\"448\">q=t/ell</text><text x=\"12\" y=\"24\">"
           << fieldName << "</text>";
    for (std::size_t index = 0; index < series.size(); ++index) {
        const CollisionCurveSeries& item = series[index];
        const std::string& color = colors[index % colors.size()];
        output << "<polyline fill=\"none\" stroke=\"" << color
               << "\" stroke-width=\"2\" points=\"";
        for (const CollisionCurvePoint& point : item.points) {
            if (field == CollisionPlotField::Hazard &&
                point.atRisk < static_cast<std::uint64_t>(config.minimumRiskSetForError)) {
                continue;
            }
            const double value = field == CollisionPlotField::Survival ? point.survival :
                (field == CollisionPlotField::Hazard ? point.hazard : point.cumulativeHazard);
            if (!std::isfinite(value)) continue;
            const double x = 70.0 + 690.0 * point.q / config.maximumTime;
            const double y = 400.0 - 360.0 * std::clamp(value / yMaximum, 0.0, 1.0);
            output << x << ',' << y << ' ';
        }
        output << "\"/><text x=\"520\" y=\"" << 52 + 20 * index
               << "\" fill=\"" << color << "\">" << xml(item.legendLabel)
               << "</text>";
    }
    output << "<text x=\"70\" y=\"425\">";
    if (config.collisionVisualization.comparison == "states") {
        output << "kernel: " << xml(series.front().kernel.id);
    } else {
        output << "state: " << xml(series.front().stateId);
    }
    output << "; training resolution; solid: Monte Carlo</text></svg>\n";
}

void runCollisionStateFirstPassage(const FirstPassageExperimentConfig& config) {
    std::optional<std::ofstream> samples;
    if (config.writeRawSamples) {
        samples.emplace(openOutput(config.outputDirectory / "first_passage_samples.csv"));
        *samples << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,"
                    "state_id,beta_0,beta_a,beta_g,max_q,"
                    "base_step,training_resolution,trajectory,event_q,event_distance,event,censored,"
                    "crossing_slope,crossing_derivative,seed,integrator_tolerance,minimum_step,"
                    "bridge_sigma_margin,transitions,bridge_refinements,deepest_refinement\n";
    }
    std::ofstream summary = openOutput(config.outputDirectory / "first_passage_summary.csv");
    summary << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,"
               "state_id,beta_0,beta_a,beta_g,max_q,base_step,"
               "training_resolution,trajectories,events,censored,event_probability,"
               "restricted_mean_q,event_q10,event_q50,event_q90,crossing_slope_q10,"
               "crossing_slope_q50,crossing_slope_q90,mean_transitions,mean_bridge_refinements\n";
    std::ofstream curves = openOutput(config.outputDirectory / "first_passage_curves.csv");
    curves << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,state_id,"
              "beta_0,beta_a,beta_g,base_step,"
              "training_resolution,bin,q_begin,q_end,at_risk,events,survival,"
              "first_passage_density,hazard_mc,cumulative_hazard_nelson_aalen,"
              "cumulative_hazard_product_limit\n";

    const double trainingStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    std::vector<CollisionCurveSeries> plottedSeries;
    for (std::size_t kernelIndex = 0; kernelIndex < config.kernels.size(); ++kernelIndex) {
        const FirstPassageKernelConfig& kernel = config.kernels[kernelIndex];
        for (std::size_t stateIndex = 0; stateIndex < config.collisionStates.size(); ++stateIndex) {
            const FirstPassageCollisionState& state = config.collisionStates[stateIndex];
            for (std::size_t resolution = 0; resolution < config.stepSizes.size(); ++resolution) {
                const double step = config.stepSizes[resolution];
                const bool trainingResolution =
                    std::abs(step - trainingStep) <= 1e-12 * trainingStep;
                const std::vector<CollisionSample> generated = generateCollisionSamples(
                    config, kernel, state, kernelIndex, stateIndex, resolution, step);
                if (samples) {
                    writeCollisionSamples(*samples, config, kernel, state, step,
                                          trainingResolution, generated);
                }
                writeCollisionSummary(summary, config, kernel, state, step,
                                      trainingResolution, generated);
                std::vector<CollisionCurvePoint> curvePoints = writeCollisionCurves(
                    curves, config, kernel, state, step, trainingResolution, generated);
                const bool compareStates =
                    config.collisionVisualization.comparison == "states";
                const std::string selectedKernel =
                    config.collisionVisualization.kernelId.value_or("");
                const std::string selectedState =
                    config.collisionVisualization.stateIds.empty()
                    ? config.collisionStates.front().id
                    : config.collisionVisualization.stateIds.front();
                const bool selectedForPlot = trainingResolution &&
                    (compareStates
                        ? kernel.id == selectedKernel &&
                          std::find(config.collisionVisualization.stateIds.begin(),
                                    config.collisionVisualization.stateIds.end(),
                                    state.id) != config.collisionVisualization.stateIds.end()
                        : state.id == selectedState);
                if (selectedForPlot) {
                    plottedSeries.push_back(
                        {kernel, state.id, compareStates ? state.id : kernel.id,
                         std::move(curvePoints)});
                }
            }
        }
    }
    if (config.collisionVisualization.comparison == "states") {
        const auto& order = config.collisionVisualization.stateIds;
        std::stable_sort(plottedSeries.begin(), plottedSeries.end(),
            [&](const CollisionCurveSeries& left, const CollisionCurveSeries& right) {
                const auto leftPosition = std::find(order.begin(), order.end(), left.stateId);
                const auto rightPosition = std::find(order.begin(), order.end(), right.stateId);
                return leftPosition < rightPosition;
            });
    }
    writeCollisionSvg(config, plottedSeries, CollisionPlotField::Survival);
    writeCollisionSvg(config, plottedSeries, CollisionPlotField::Hazard);
    writeCollisionSvg(config, plottedSeries, CollisionPlotField::CumulativeHazard);
}

} // namespace

void runFirstPassageExperiment(const FirstPassageExperimentConfig& config) {
    std::filesystem::create_directories(config.outputDirectory);
    writeResolvedFirstPassageConfig(config,
        config.outputDirectory / "resolved_first_passage_config.json");
    if (config.initialConditionType == "collision_state") {
        runCollisionStateFirstPassage(config);
        return;
    }
    const double fineStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    const std::size_t fineIntervals = static_cast<std::size_t>(
        std::llround(config.maximumTime / fineStep));
    const std::size_t sampleCount = fineIntervals + 1;
    std::vector<CurveSeries> allSeries;
    std::vector<KernelStateRecords> allStateRecords;

    for (std::size_t kernelIndex = 0; kernelIndex < config.kernels.size(); ++kernelIndex) {
        const FirstPassageKernelConfig& kernelConfig = config.kernels[kernelIndex];
        const StationaryKernel kernel(kernelConfig);
        CirculantSampler diagnosticSampler(kernel, sampleCount, fineStep,
                                           config.maximumEmbeddingExpansions);
        std::vector<CurveSeries> kernelSeries;
        for (const double step : config.stepSizes) {
            CurveSeries series;
            series.kernel = kernelConfig;
            series.step = step;
            series.embeddingSize = diagnosticSampler.embeddingSize();
            series.minimumEigenvalue = diagnosticSampler.minimumEigenvalue();
            series.eventTimes.resize(static_cast<std::size_t>(config.trajectories));
            kernelSeries.push_back(std::move(series));
        }
        KernelStateRecords stateRecords;
        stateRecords.kernel = kernelConfig;
        std::vector<std::size_t> snapshotIndices;
        if (config.stateAnalysis.enabled) {
            stateRecords.records.reserve(static_cast<std::size_t>(config.trajectories) *
                                         config.stateAnalysis.snapshotTimes.size());
            for (const double requested : config.stateAnalysis.snapshotTimes) {
                const std::size_t index = static_cast<std::size_t>(std::llround(requested / fineStep));
                snapshotIndices.push_back(index);
                stateRecords.actualSnapshotTimes.push_back(static_cast<double>(index) * fineStep);
            }
        }

        unsigned int workerCount = config.threadCount > 0
            ? static_cast<unsigned int>(config.threadCount) : std::thread::hardware_concurrency();
        workerCount = std::max(1u, std::min(workerCount,
            static_cast<unsigned int>(config.trajectories)));
        // Each worker owns FFT input/output/spectrum buffers. Bound automatic
        // parallelism for slowly decaying kernels whose positive embedding may
        // be much larger than the requested sample window.
        constexpr std::size_t memoryBudget = 512ULL * 1024ULL * 1024ULL;
        const std::size_t estimatedWorkerBytes = diagnosticSampler.embeddingSize() *
            (3 * sizeof(std::complex<double>) + sizeof(double));
        const unsigned int memoryWorkers = static_cast<unsigned int>(std::max<std::size_t>(
            1, memoryBudget / std::max<std::size_t>(1, estimatedWorkerBytes)));
        workerCount = std::min(workerCount, memoryWorkers);
        std::vector<std::vector<StateRecord>> workerRecords(workerCount);
        std::vector<std::exception_ptr> workerErrors(workerCount);
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        for (unsigned int worker = 0; worker < workerCount; ++worker) {
            const int begin = static_cast<int>(
                static_cast<std::uint64_t>(config.trajectories) * worker / workerCount);
            const int end = static_cast<int>(
                static_cast<std::uint64_t>(config.trajectories) * (worker + 1) / workerCount);
            workers.emplace_back([&, worker, begin, end]() {
                try {
                    CirculantSampler sampler(kernel, sampleCount, fineStep,
                                             config.maximumEmbeddingExpansions);
                    workerRecords[worker].reserve(static_cast<std::size_t>(end - begin) *
                                                  snapshotIndices.size());
                    for (int trajectory = begin; trajectory < end; ++trajectory) {
                        std::uint64_t seed = config.seed ^
                            (0x9e3779b97f4a7c15ULL * (kernelIndex + 1)) ^
                            (0xbf58476d1ce4e5b9ULL * (static_cast<std::uint64_t>(trajectory) + 1));
                        seed ^= seed >> 30;
                        seed *= 0xbf58476d1ce4e5b9ULL;
                        seed ^= seed >> 27;
                        seed *= 0x94d049bb133111ebULL;
                        seed ^= seed >> 31;
                        Random rng(seed);
                        const std::vector<double> path = sampler.sampleConditionedValue(
                            rng, config.processMean, config.initialValue);
                        double fineCrossing = std::numeric_limits<double>::infinity();
                        for (std::size_t seriesIndex = 0;
                             seriesIndex < kernelSeries.size(); ++seriesIndex) {
                            const std::size_t stride = static_cast<std::size_t>(
                                std::llround(kernelSeries[seriesIndex].step / fineStep));
                            const double crossing = crossingTime(
                                path, stride, fineStep, config.threshold, config.maximumTime);
                            kernelSeries[seriesIndex].eventTimes[trajectory] = crossing;
                            if (stride == 1) fineCrossing = crossing;
                        }
                        for (std::size_t snapshot = 0;
                             snapshot < snapshotIndices.size(); ++snapshot) {
                            const std::size_t index = snapshotIndices[snapshot];
                            const double time = stateRecords.actualSnapshotTimes[snapshot];
                            if (!(fineCrossing > time)) continue;
                            StateRecord record;
                            record.trajectory = static_cast<std::uint64_t>(trajectory);
                            record.snapshot = snapshot;
                            record.value = path[index];
                            record.derivative =
                                (path[index + 1] - path[index - 1]) / (2.0 * fineStep);
                            record.eventInWindow =
                                fineCrossing <= time + config.stateAnalysis.futureWindow;
                            workerRecords[worker].push_back(record);
                        }
                    }
                } catch (...) {
                    workerErrors[worker] = std::current_exception();
                }
            });
        }
        for (std::thread& worker : workers) worker.join();
        for (const std::exception_ptr& error : workerErrors) {
            if (error) std::rethrow_exception(error);
        }
        for (std::vector<StateRecord>& records : workerRecords) {
            stateRecords.records.insert(stateRecords.records.end(),
                                        records.begin(), records.end());
        }
        const std::vector<RiceCurvePoint> riceCurve = makeRiceCurve(config, kernel);
        for (CurveSeries& series : kernelSeries) {
            series.points = makeCurve(config, kernel, series.eventTimes, riceCurve);
            allSeries.push_back(std::move(series));
        }
        if (config.stateAnalysis.enabled) allStateRecords.push_back(std::move(stateRecords));
    }

    computeSeriesSummaries(config, allSeries);
    writeCurveCsv(config, allSeries);
    writeSummaryCsv(config, allSeries);
    writeRawSamples(config, allSeries);
    writeStateOutputs(config, allStateRecords);
    writeSvg(config, allSeries, false);
    writeSvg(config, allSeries, true);
    writeRiceDensitySvg(config, allSeries);
}

} // namespace mf

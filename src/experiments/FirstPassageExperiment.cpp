#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/FirstPassageRice.h"
#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/gpss/RayMeanProfile.h"
#include "macrofacet/gpss/Matern32Reference.h"

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
#include <sstream>
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

std::string preciseString(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
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

    std::vector<double> samplePositiveExterior(Random& rng, double birthMean,
                                               double& birthDerivative) {
        const double sigma = std::sqrt(kernel_.config().variance);
        const double b = birthMean / sigma;
        // Draw the unknown birth value conditional only on F(0)>0. The
        // derivative remains random, unlike a prescribed surface birth.
        const double z = -normalQuantileFromLogCdf(normalLogCdf(b) + std::log(rng.openUniform01()));
        const double value = std::max(sigma*z,
            std::nextafter(-birthMean,std::numeric_limits<double>::infinity()));
        auto result = sampleConditionedValue(rng,0.0,value);
        const double negativeGhost = time_[embeddingSize_-1].real() +
            kernel_.covariance(step_) / kernel_.config().variance * (value-time_[0].real());
        birthDerivative = (result[1]-negativeGhost)/(2.0*step_);
        return result;
    }

    std::size_t embeddingSize() const { return embeddingSize_; }
    double minimumEigenvalue() const { return minimumEigenvalue_; }

    void prepareFixedEndpoint(std::size_t index) {
        if (index < 2 || index + 1 >= sampleCount_) {
            throw std::invalid_argument("fixed endpoint requires at least two intervals and a residual ghost node");
        }
        endpointIndex_ = index;
        endpointVariance_ = startConditionedCovariance(index, index);
        if (!std::isfinite(endpointVariance_) || endpointVariance_ <=
            256.0 * std::numeric_limits<double>::epsilon() * kernel_.config().variance) {
            throw std::runtime_error("fixed endpoint conditional variance is numerically degenerate; "
                                     "use a larger target distance (no jitter was added)");
        }
        endpointRatios_.resize(sampleCount_);
        for (std::size_t i = 0; i < sampleCount_; ++i) {
            endpointRatios_[i] = startConditionedCovariance(i, index) / endpointVariance_;
        }
        endpointRatios_[0] = 0.0;
        endpointRatios_[index] = 1.0;
        negativeGhostRatio_ = startConditionedCovariance(embeddingSize_ - 1, index) / endpointVariance_;
    }

    double endpointVariance() const { return endpointVariance_; }

    double startConditionedMean(std::size_t index, double initialValue,
                                 double initialDerivative) const {
        return circulantCovariance(index, 0) / kernel_.config().variance * initialValue +
               birthDerivativeCovariance(index) / birthDerivativeVariance() * initialDerivative;
    }

    std::pair<double, double> bridgeSlopeMoments(double initialValue,
        double initialDerivative, double endpointValue) const {
        const auto plus = endpointIndex_ + 1, minus = endpointIndex_ - 1;
        const double mean = (startConditionedMean(plus, initialValue, initialDerivative) -
                             startConditionedMean(minus, initialValue, initialDerivative)) / (2.0 * step_);
        const double covariance = (startConditionedCovariance(plus, endpointIndex_) -
            startConditionedCovariance(minus, endpointIndex_)) / (2.0 * step_);
        const double variance = (startConditionedCovariance(plus, plus) +
            startConditionedCovariance(minus, minus) -
            2.0 * startConditionedCovariance(plus, minus)) / (4.0 * step_ * step_);
        const double conditionalVariance = variance - covariance * covariance / endpointVariance_;
        if (conditionalVariance < -1e-9 * kernel_.derivativeVariance()) {
            throw std::runtime_error("fixed endpoint slope covariance is not positive semidefinite");
        }
        return {mean + covariance / endpointVariance_ * (endpointValue -
            startConditionedMean(endpointIndex_, initialValue, initialDerivative)),
            std::sqrt(std::max(0.0, conditionalVariance))};
    }

    std::vector<double> sampleConditionedEndpoint(Random& rng, double initialValue,
        double initialDerivative, double endpointValue, double& birthSlopeError) {
        if (endpointRatios_.empty()) throw std::logic_error("fixed endpoint conditioner was not prepared");
        auto result = sampleConditionedValueDerivative(rng, initialValue, initialDerivative);
        const double derivativeDraw = (time_[1].real() - time_[embeddingSize_ - 1].real()) / (2.0 * step_);
        const auto negative = embeddingSize_ - 1;
        double negativeGhost = time_[negative].real() +
            circulantCovariance(negative, 0) / kernel_.config().variance * (initialValue - time_[0].real()) +
            birthDerivativeCovariance(negative) / birthDerivativeVariance() * (initialDerivative - derivativeDraw);
        const double innovation = endpointValue - result[endpointIndex_];
        for (std::size_t i = 0; i < result.size(); ++i) result[i] += endpointRatios_[i] * innovation;
        negativeGhost += negativeGhostRatio_ * innovation;
        birthSlopeError = (result[1] - negativeGhost) / (2.0 * step_) - initialDerivative;
        result[0] = initialValue;
        result[endpointIndex_] = endpointValue;
        return result;
    }

private:
    double birthDerivativeVariance() const {
        return (kernel_.config().variance - circulantCovariance(1, embeddingSize_ - 1)) /
               (2.0 * step_ * step_);
    }

    double birthDerivativeCovariance(std::size_t index) const {
        return (circulantCovariance(index, 1) - circulantCovariance(index, embeddingSize_ - 1)) /
               (2.0 * step_);
    }

    double startConditionedCovariance(std::size_t first, std::size_t second) const {
        return circulantCovariance(first, second) -
            circulantCovariance(first, 0) * circulantCovariance(second, 0) / kernel_.config().variance -
            birthDerivativeCovariance(first) * birthDerivativeCovariance(second) / birthDerivativeVariance();
    }
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
    std::size_t endpointIndex_ = 0;
    double endpointVariance_ = 0.0;
    double negativeGhostRatio_ = 0.0;
    std::vector<double> endpointRatios_;
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

struct CollisionSample {
    double eventQ = 0.0;
    bool event = false;
    double crossingSlope = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t seed = 0;
    std::uint64_t transitions = 0;
    std::uint64_t bridgeRefinements = 0;
    int deepestRefinement = 0;
};

struct MeanProfileSegment {
    std::size_t gridIndex = 0;
    double qBegin = 0.0;
    double qEnd = 0.0;
    double fractionBegin = 0.0;
    double fractionEnd = 1.0;
    HermitePolynomial mean;
};

// Geometry is shared by the reference sampler and future neural segmentation.
RayMeanProfile rayMeanProfile(const FirstPassageExperimentConfig& config,
    const FirstPassageKernelConfig& kernel, const FirstPassageCollisionState& state,
    double maximumStep) {
    if (!config.processMeanField)
        return RayMeanProfile::affine(state.beta0,state.betaMeanSlope,config.maximumTime,maximumStep);
    const auto& ray=*state.ray;
    return RayMeanProfile::fromField(*config.processMeanField,ray.origin,ray.direction,
        std::sqrt(kernel.variance),kernel.lengthScale,
        config.maximumTime*kernel.lengthScale,maximumStep);
}

// Adapter for the existing uniform circulant residual and fixed-endpoint tools.
// The common profile owns the cubic; this only splits it at residual grid nodes.
std::vector<MeanProfileSegment> prepareMeanProfile(
    const FirstPassageExperimentConfig& config, const FirstPassageKernelConfig& kernel,
    const FirstPassageCollisionState& state, double step, bool includeAffine = false) {
    std::vector<MeanProfileSegment> segments;
    if (!config.processMeanField && !includeAffine) return segments;
    const auto profile=rayMeanProfile(config,kernel,state,config.profileMaximumStep);
    const auto intervals=static_cast<std::size_t>(std::llround(config.maximumTime/step));
    std::vector<double> knots{0.0};
    for (const auto& segment : profile.segments()) knots.push_back(segment.end);
    for (std::size_t i=1; i<intervals; ++i) knots.push_back(i*step);
    std::sort(knots.begin(),knots.end());
    knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
    std::size_t cell=0;
    for (std::size_t i=1; i<knots.size(); ++i) {
        const double lo=knots[i-1],hi=knots[i];
        while (profile.segments()[cell].end<=lo) ++cell;
        const auto part=profile.segments()[cell].restricted(lo,hi);
        const auto index=std::min(intervals-1,static_cast<std::size_t>(std::floor((0.5*lo+0.5*hi)/step)));
        segments.push_back({index,lo,hi,(lo-index*step)/step,(hi-index*step)/step,part.polynomial});
    }
    return segments;
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

CollisionSample sampleGridCollisionFirstPassage(
    CirculantSampler& sampler, const FirstPassageCollisionState& parameter,
    double maximumQ, double step, std::uint64_t seed,
    const std::vector<MeanProfileSegment>& profile, bool positiveExterior = false) {
    CollisionSample result;
    result.eventQ = maximumQ;
    result.seed = seed;
    Random rng(seed);
    double birthDerivative = parameter.betaCollisionSlope - parameter.betaMeanSlope;
    const std::vector<double> residual = positiveExterior
        ? sampler.samplePositiveExterior(rng,parameter.beta0,birthDerivative)
        : sampler.sampleConditionedValueDerivative(rng,-parameter.beta0,birthDerivative);
    result.transitions = residual.size() - 1;
    if (!profile.empty()) {
        // Add the complete, cell-wise mean to a Hermite interpolation of the
        // residual. Do not interpolate across the mean's derivative jumps.
        for (const auto& segment : profile) {
            const std::size_t i = segment.gridIndex;
            const double slope0 = i == 0
                ? birthDerivative
                : (residual[i + 1] - residual[i - 1]) / (2.0 * step);
            const double slope1 = i + 2 < residual.size()
                ? (residual[i + 2] - residual[i]) / (2.0 * step)
                : (residual[i + 1] - residual[i]) / step;
            const auto stochastic = hermitePolynomial(residual[i], slope0,
                residual[i + 1], slope1, step);
            const double width = segment.qEnd - segment.qBegin;
            const auto part = hermitePolynomial(
                stochastic.value(segment.fractionBegin),
                stochastic.derivative(segment.fractionBegin) / step,
                stochastic.value(segment.fractionEnd),
                stochastic.derivative(segment.fractionEnd) / step, width);
            HermitePolynomial total{part.a + segment.mean.a, part.b + segment.mean.b,
                                    part.c + segment.mean.c, part.d + segment.mean.d};
            // The prescribed birth is not itself a new collision.
            if (segment.qBegin == 0.0 && !positiveExterior) {
                total.d = 0.0;
                total.c = width * parameter.betaCollisionSlope;
            }
            const auto crossing = firstHermiteDowncrossing(total, width, 1e-12 * width);
            if (crossing.found) {
                result.event = true;
                result.eventQ = segment.qBegin + crossing.fraction * width;
                result.crossingSlope = std::max(0.0, -crossing.derivative);
                return result;
            }
        }
        return result;
    }
    std::vector<double> values(residual.size());
    values[0] = positiveExterior ? parameter.beta0+residual[0] : 0.0;
    for (std::size_t index = 1; index < residual.size(); ++index) {
        const double q = static_cast<double>(index) * step;
        values[index] = parameter.beta0 + parameter.betaMeanSlope * q +
                        residual[index];
    }
    for (std::size_t index = 1; index < values.size(); ++index) {
        const double derivative0 = index == 1
            ? (positiveExterior ? parameter.betaMeanSlope+birthDerivative : parameter.betaCollisionSlope)
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

Json kernelJson(const FirstPassageKernelConfig& kernel) {
    Json result{{"id", kernel.id}, {"type", kernel.type},
                {"variance", kernel.variance}, {"length_scale", kernel.lengthScale}};
    if (kernel.type == "rational_quadratic") result["alpha"] = kernel.alpha;
    if (kernel.type == "matern_3_2") result["parameterization"] = "unit_decay";
    if (kernel.type == "squared_exponential") result["parameterization"] = "unit_length";
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

Vector3 firstPassageVector(const Json& source, const std::string& name) {
    if (!source.is_array() || source.size() != 3) {
        throw std::invalid_argument(name + " requires three coordinates");
    }
    Vector3 result(source[0].get<double>(), source[1].get<double>(), source[2].get<double>());
    if (!result.allFinite()) throw std::invalid_argument(name + " must be finite");
    return result;
}

void validateMeanFieldRays(const FirstPassageExperimentConfig& config) {
    if (!config.processMeanField) return;
    if (config.initialConditionType == "fixed_value") {
        throw std::invalid_argument("process.mean_field requires collision_state or positive_exterior with rays");
    }
    for (const auto& state : config.collisionStates) {
        if (!state.ray) throw std::invalid_argument("mean_field requires physical ray conditions");
        const auto& ray = *state.ray;
        if (!ray.origin.allFinite() || !ray.direction.allFinite() || !ray.gradient.allFinite() ||
            std::abs(ray.direction.norm() - 1.0) > 1e-10 ||
            (config.initialConditionType=="collision_state" && !(ray.gradient.dot(ray.direction) > 0.0))) {
            throw std::invalid_argument("ray " + state.id +
                " requires finite coordinates, a unit direction and gradient.dot(direction)>0");
        }
        for (const auto& kernel : config.kernels) {
            if (const auto stored = config.processMeanField->intrinsicSigma()) {
                // The bake stores float32 sigma. Restating it in a JSON double
                // must agree at the grid's own precision.
                if (static_cast<float>(*stored) != static_cast<float>(std::sqrt(kernel.variance))) {
                    throw std::invalid_argument("kernel variance disagrees with baked SDF sigma; "
                                                "re-bake or use its stored sigma");
                }
            }
            config.processMeanField->requireFullRayCoverage(ray.origin, ray.direction,
                config.maximumTime * kernel.lengthScale);
        }
        const auto jet = config.processMeanField->evaluate(ray.origin);
        requireFinite(jet.value, "birth mean value");
        if (!jet.gradient.allFinite()) throw std::invalid_argument("birth mean gradient is not finite");
    }
}

void validateFixedEndpoints(const FirstPassageExperimentConfig& config) {
    const auto& fixed = config.fixedEndpoint;
    if (fixed.only && !fixed.enabled) throw std::invalid_argument("fixed_endpoint.only requires enabled=true");
    if (!fixed.enabled) return;
    if (config.initialConditionType != "collision_state" || fixed.distances.empty() ||
        fixed.trajectories < 0) {
        throw std::invalid_argument("fixed_endpoint requires collision_state, nonempty distances and trajectories>=0");
    }
    for (double distance : fixed.distances) {
        requirePositive(distance, "fixed_endpoint.distances");
        for (const auto& kernel : config.kernels) {
            const double q = distance / kernel.lengthScale;
            if (q > config.maximumTime + 1e-10 * config.maximumTime) {
                throw std::invalid_argument("fixed endpoint exceeds grid.max_time * kernel.length_scale");
            }
            for (double step : config.stepSizes) {
                if (q / step < 2.0 - 1e-9 || !closeInteger(q / step)) {
                    throw std::invalid_argument("each fixed endpoint distance / ell must be a grid node "
                        "at least two steps after birth for every configured kernel and resolution");
                }
            }
        }
    }
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
        if (process.contains("mean_field")) {
            Json field = process.at("mean_field");
            // Material grids do not define the covariance of this global GP.
            field["use_alpha_grid"] = false;
            config.processMeanField = buildMeanFromJson(field).mean;
            config.meanFieldConfiguration = field.dump();
        }
    }
    if (experiment.contains("initial_condition")) {
        const Json& initial = experiment.at("initial_condition");
        const std::string type = initial.value("type", "fixed_value");
        config.initialConditionType = type;
        if (type == "fixed_value") {
            config.initialValue = initial.at("value").get<double>();
        } else if (type == "collision_state" || type == "positive_exterior") {
            if (config.processMeanField) {
                if (initial.contains("parameter_space") || !initial.contains("rays") ||
                    !initial.at("rays").is_array() || initial.at("rays").empty()) {
                    throw std::invalid_argument("mean_field requires nonempty initial_condition.rays "
                                                "instead of parameter_space");
                }
                for (const auto& source : initial.at("rays")) {
                    FirstPassageRayCondition ray;
                    ray.origin = firstPassageVector(source.at("origin"), "ray.origin");
                    ray.direction = normalizedOrThrow(
                        firstPassageVector(source.at("direction"), "ray.direction"));
                    if (type=="collision_state")
                        ray.gradient = firstPassageVector(source.at("gradient"), "ray.gradient");
                    else if (source.contains("gradient"))
                        throw std::invalid_argument("positive_exterior must not specify an observed gradient");
                    FirstPassageCollisionState state;
                    state.id = source.at("id").get<std::string>();
                    state.ray = ray;
                    config.collisionStates.push_back(std::move(state));
                }
            } else {
                if (initial.contains("rays")) {
                    throw std::invalid_argument("physical rays require process.mean_field");
                }
                if (type=="collision_state") {
                    config.collisionStates = parseCollisionStates(initial, config.seed);
                } else {
                    if (!initial.contains("profiles") || !initial.at("profiles").is_array() ||
                        initial.at("profiles").empty() || initial.contains("parameter_space"))
                        throw std::invalid_argument("positive_exterior requires nonempty affine profiles or mean_field rays");
                    for (const auto& source : initial.at("profiles")) {
                        if (source.contains("beta_g") || source.contains("value") || source.contains("gradient"))
                            throw std::invalid_argument("positive_exterior profiles contain only known mean information");
                        FirstPassageCollisionState state;
                        state.id=source.at("id").get<std::string>();
                        state.beta0=source.at("beta_0").get<double>();
                        state.betaMeanSlope=source.at("beta_a").get<double>();
                        state.betaCollisionSlope=0.0;
                        config.collisionStates.push_back(state);
                    }
                }
            }
            // These legacy diagnostics have different conditioning semantics.
            config.riceSeries.enabled = false;
            config.stateAnalysis.enabled = false;
            config.writeRawSamples = true;
        } else {
            throw std::invalid_argument(
                "first_passage.initial_condition.type must be fixed_value, collision_state or positive_exterior");
        }
    }
    if (experiment.contains("grid")) {
        const Json& grid = experiment.at("grid");
        config.maximumTime = grid.value("max_time", config.maximumTime);
        if (grid.contains("step_sizes")) {
            config.stepSizes = grid.at("step_sizes").get<std::vector<double>>();
        }
    }
    if (experiment.contains("profile"))
        config.profileMaximumStep=experiment.at("profile").value("maximum_step",config.profileMaximumStep);
    requirePositive(config.profileMaximumStep,"profile.maximum_step");
    if (experiment.contains("curve")) {
        config.curveBins = experiment.at("curve").value("bins", config.curveBins);
    }
    if (experiment.contains("fixed_endpoint")) {
        const auto& fixed = experiment.at("fixed_endpoint");
        config.fixedEndpoint.enabled = fixed.value("enabled", false);
        config.fixedEndpoint.only = fixed.value("only", false);
        config.fixedEndpoint.trajectories = fixed.value("trajectories", 0);
        config.fixedEndpoint.distances = fixed.value("distances", std::vector<double>{});
        std::sort(config.fixedEndpoint.distances.begin(), config.fixedEndpoint.distances.end());
        config.fixedEndpoint.distances.erase(std::unique(config.fixedEndpoint.distances.begin(),
            config.fixedEndpoint.distances.end()), config.fixedEndpoint.distances.end());
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
        const std::string defaultType = config.initialConditionType != "fixed_value"
            ? "collision_state_auto" : "exact_grid_circulant";
        const std::string type = sampler.value("type", defaultType);
        if (config.initialConditionType == "fixed_value" && type != "exact_grid_circulant") {
            throw std::invalid_argument(
                "fixed_value first passage requires sampler.type=exact_grid_circulant");
        }
        if (config.initialConditionType != "fixed_value" &&
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
        if (kernel.type=="matern_3_2" && source.value("parameterization", "unit_decay")!="unit_decay")
            throw std::invalid_argument("matern_3_2 requires unit_decay: rho(x)=(1+x)exp(-x)");
        if (kernel.type=="squared_exponential" && source.value("parameterization", "unit_length")!="unit_length")
            throw std::invalid_argument("squared_exponential requires unit_length: rho(x)=exp(-x*x/2)");
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

    if (config.initialConditionType=="positive_exterior") {
        for (const auto& kernel : config.kernels)
            if (kernel.type!="matern_3_2" && kernel.type!="squared_exponential")
                throw std::invalid_argument("positive_exterior reference supports matern_3_2 and squared_exponential");
    }
    requireFinite(config.processMean, "first_passage.process.mean");
    requireFinite(config.threshold, "first_passage.process.threshold");
    if (config.initialConditionType == "fixed_value") {
        requireFinite(config.initialValue, "first_passage.initial_condition.value");
        if (!(config.initialValue > config.threshold)) {
            throw std::invalid_argument("initial_condition.value must be above process.threshold");
        }
    } else {
        // Both renewal start modes retain finite-horizon censoring. Raw samples
        // are needed for future joint distance/speed training.
        if (config.collisionStates.empty()) {
            throw std::invalid_argument("collision-state parameter space cannot be empty");
        }
        if (config.threshold != 0.0 || config.processMean != 0.0) {
            throw std::invalid_argument(
                "collision_state requires process.mean=threshold=0");
        }
        std::unordered_set<std::string> stateIds;
        for (const FirstPassageCollisionState& state : config.collisionStates) {
            if (state.id.empty() || !stateIds.insert(state.id).second) {
                throw std::invalid_argument("collision state ids must be nonempty and unique");
            }
            requireFinite(state.beta0, "collision beta_0");
            requireFinite(state.betaMeanSlope, "collision beta_a");
            if (config.initialConditionType=="collision_state")
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
    if (config.initialConditionType != "fixed_value") {
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
        if (hasMatern32 && !(config.fixedEndpoint.enabled && config.fixedEndpoint.only) &&
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
    validateFixedEndpoints(config);
    validateMeanFieldRays(config);
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
    if (config.initialConditionType != "fixed_value") {
        Json states = Json::array();
        for (const FirstPassageCollisionState& item : config.collisionStates) {
            states.push_back({{"id", item.id}, {"beta_0", item.beta0},
                              {"beta_a", item.betaMeanSlope},
                              {"beta_g", item.betaCollisionSlope}});
        }
        initial = {{"type", config.initialConditionType},
                   {"parameter_space", {{"type", "explicit"},
                                        {"states", std::move(states)}}}};
        if (config.initialConditionType=="positive_exterior") {
            Json profiles=Json::array();
            for (const auto& item : config.collisionStates)
                profiles.push_back({{"id",item.id},{"beta_0",item.beta0},{"beta_a",item.betaMeanSlope}});
            initial={{"type",config.initialConditionType},{"profiles",std::move(profiles)}};
        }
        if (config.processMeanField) {
            Json rays = Json::array();
            for (const auto& item : config.collisionStates) {
                const auto& ray = *item.ray;
                rays.push_back({{"id", item.id},
                    {"origin", {ray.origin.x(), ray.origin.y(), ray.origin.z()}},
                    {"direction", {ray.direction.x(), ray.direction.y(), ray.direction.z()}},
                    {"gradient", {ray.gradient.x(), ray.gradient.y(), ray.gradient.z()}}});
            }
            if (config.initialConditionType=="positive_exterior")
                for (auto& ray : rays) ray.erase("gradient");
            initial = {{"type", config.initialConditionType}, {"rays", std::move(rays)}};
        }
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
            {"profile", {{"maximum_step",config.profileMaximumStep}}},
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
    if (config.fixedEndpoint.enabled) {
        root["first_passage"]["fixed_endpoint"] = {
            {"enabled", true}, {"only", config.fixedEndpoint.only},
            {"distances", config.fixedEndpoint.distances},
            {"trajectories", config.fixedEndpoint.trajectories},
            {"effective_trajectories", config.fixedEndpoint.trajectories > 0
                ? config.fixedEndpoint.trajectories : config.trajectories},
            {"sampler", "start_and_endpoint_conditioned_grid_circulant"},
            {"endpoint_value", 0.0}, {"derivative_condition", "central_difference_residual"},
            {"survival_test", "cellwise_cubic_open_interval_positivity"},
            {"histogram_weight", "survived_to_endpoint * max(0,-endpoint_derivative)"}};
    }
    if (config.processMeanField) {
        auto field = config.meanFieldConfiguration.empty()
            ? Json{{"mean_type", config.processMeanField->typeName()}}
            : Json::parse(config.meanFieldConfiguration);
        if (const auto metadata = writeMeanToJson(*config.processMeanField)) {
            field.update(*metadata);
        }
        // The shared field writer defaults to renderer material grids; this
        // experiment deliberately uses only the SDF with a stationary kernel.
        field["use_alpha_grid"] = false;
        root["first_passage"]["process"]["mean_field"] = std::move(field);
        root["mean_profile"] = {{"type", "full_field"},
            {"ray_coverage_validated", true},
            {"start_mode",config.initialConditionType},
            {"sampler", "kernel_dependent_see_sample_rows"},
            {"derivative_condition", "matern32_exact_other_kernels_central_difference"},
            {"mean_interpolation", "cellwise_cubic_along_ray"}};
    }
    std::ofstream output = openOutput(path);
    output << std::setw(2) << root << '\n';
}

namespace {

const char* collisionSamplerName(const FirstPassageExperimentConfig&,
                                 const FirstPassageKernelConfig& kernel) {
    return kernel.type == "matern_3_2"
        ? "matern32_state_space" : "conditioned_grid_circulant";
}

const char* collisionSlopeMethod(const FirstPassageExperimentConfig& config,
                                 const FirstPassageKernelConfig& kernel) {
    if (config.processMeanField) return kernel.type == "matern_3_2"
        ? "state_bridge_hermite_cellwise_mean" : "grid_residual_hermite_cellwise_mean";
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
    const bool stateSpace = kernel.type == "matern_3_2";
    const auto profile = stateSpace ? std::vector<MeanProfileSegment>{}
                                   : prepareMeanProfile(config, kernel, state, step);
    const auto referenceProfile=stateSpace ? std::make_optional(rayMeanProfile(config,kernel,state,config.profileMaximumStep))
                                           : std::nullopt;
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
                if (!stateSpace) {
                    gridSampler.emplace(dimensionlessKernel, gridSamples, step,
                                        config.maximumEmbeddingExpansions);
                }
                for (int trajectory = begin; trajectory < end; ++trajectory) {
                    const std::uint64_t seed = collisionSeed(
                        config.seed, kernelIndex, stateIndex, resolutionIndex,
                        static_cast<std::uint64_t>(trajectory));
                    if (stateSpace) {
                        Random rng(seed);
                        const RayStartCondition start{
                            config.initialConditionType=="positive_exterior"
                                ? RayStartMode::PositiveExterior : RayStartMode::SurfaceOutward,
                            state.betaCollisionSlope};
                        const auto& settings=config.collisionSampler;
                        const auto sample=sampleMatern32FirstPassage(*referenceProfile,start,
                            {step,settings.minimumStep,settings.crossingTolerance,
                             settings.bridgeSigmaMargin,settings.maximumRefinementDepth},rng);
                        result[static_cast<std::size_t>(trajectory)]={sample.distance,sample.hit,
                            sample.speed,seed,sample.transitions,sample.bridgeRefinements,
                            sample.deepestRefinement};
                    } else {
                        result[static_cast<std::size_t>(trajectory)]=sampleGridCollisionFirstPassage(
                            *gridSampler,state,config.maximumTime,step,seed,profile,
                            config.initialConditionType=="positive_exterior");
                    }
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
               << collisionSamplerName(config, kernel) << ',' << collisionSlopeMethod(config, kernel) << ','
               << sigma << ',' << kernel.lengthScale << ',' << kernel.alpha << ','
               << csv(state.id) << ','
               << state.beta0 << ',' << state.betaMeanSlope << ','
               << (config.initialConditionType=="positive_exterior" ? std::string{} : preciseString(state.betaCollisionSlope)) << ',' << config.maximumTime << ','
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
               << sample.deepestRefinement;
        if (config.processMeanField) output << ",full_field";
        output << '\n';
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
           << collisionSamplerName(config, kernel) << ',' << collisionSlopeMethod(config, kernel) << ','
           << std::sqrt(kernel.variance) << ',' << kernel.lengthScale << ','
           << kernel.alpha << ','
           << csv(state.id) << ',' << state.beta0
           << ',' << state.betaMeanSlope << ',' << (config.initialConditionType=="positive_exterior" ? std::string{} : preciseString(state.betaCollisionSlope)) << ','
           << config.maximumTime << ',' << step << ',' << (trainingResolution ? 1 : 0)
           << ',' << samples.size() << ','
           << eventTimes.size() << ',' << samples.size() - eventTimes.size() << ','
           << eventTimes.size() / count << ',' << restrictedMean / count << ','
           << sampleQuantile(eventTimes, 0.1) << ',' << sampleQuantile(eventTimes, 0.5)
           << ',' << sampleQuantile(eventTimes, 0.9) << ','
           << sampleQuantile(slopes, 0.1) << ',' << sampleQuantile(slopes, 0.5) << ','
           << sampleQuantile(slopes, 0.9) << ',' << transitions / count << ','
           << refinements / count;
    if (config.processMeanField) output << ",full_field";
    output << '\n';
}

void writeCollisionCurves(std::ofstream& output,
                          const FirstPassageExperimentConfig& config,
                          const FirstPassageKernelConfig& kernel,
                          const FirstPassageCollisionState& state,
                          double step, bool trainingResolution,
                          const std::vector<CollisionSample>& samples) {
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
        output << csv(kernel.id) << ',' << kernel.type << ','
               << collisionSamplerName(config, kernel) << ',' << collisionSlopeMethod(config, kernel) << ','
               << std::sqrt(kernel.variance) << ',' << kernel.lengthScale << ','
               << kernel.alpha << ','
               << csv(state.id) << ','
               << state.beta0 << ',' << state.betaMeanSlope << ','
               << (config.initialConditionType=="positive_exterior" ? std::string{} : preciseString(state.betaCollisionSlope)) << ',' << step << ','
               << (trainingResolution ? 1 : 0) << ',' << bin << ',' << begin << ','
               << end << ',' << atRisk << ',' << events << ',' << survival << ','
               << density << ','
               << hazard << ',' << nelsonAalen << ',' << cumulativeHazard;
        if (config.processMeanField) output << ",full_field";
        output << '\n';
    }
}

FirstPassageCollisionState resolveCollisionState(const FirstPassageExperimentConfig& config,
    const FirstPassageKernelConfig& kernel, FirstPassageCollisionState state) {
    if (config.processMeanField) {
        const auto& ray = *state.ray;
        const auto jet = config.processMeanField->evaluate(ray.origin);
        const double sigma = std::sqrt(kernel.variance);
        state.beta0 = jet.value / sigma;
        state.betaMeanSlope = rayMeanProfile(config,kernel,state,config.profileMaximumStep)
            .segments().front().derivative(0.0);
        state.betaCollisionSlope = kernel.lengthScale * ray.gradient.dot(ray.direction) / sigma;
    }
    return state;
}

struct FixedEndpointSample {
    double slope = 0.0; // signed d(F/sigma)/dq, before selection or flux weighting
    double startValue = 0.0;
    double endpointValue = 0.0;
    double birthSlopeError = 0.0;
    bool survived = true;
    std::uint64_t seed = 0;
};

FixedEndpointSample sampleFixedEndpoint(CirculantSampler& sampler,
    const FirstPassageCollisionState& state, const std::vector<MeanProfileSegment>& profile,
    std::size_t endpointIndex, double step, double endpointResidual, std::uint64_t seed) {
    FixedEndpointSample sample;
    sample.seed = seed;
    Random rng(seed);
    const auto residual = sampler.sampleConditionedEndpoint(rng, -state.beta0,
        state.betaCollisionSlope - state.betaMeanSlope, endpointResidual, sample.birthSlopeError);
    sample.startValue = residual[0] + state.beta0;
    sample.endpointValue = residual[endpointIndex] - endpointResidual;
    const auto& last = profile.back();
    sample.slope = (residual[endpointIndex + 1] - residual[endpointIndex - 1]) / (2.0 * step) +
        last.mean.derivative(1.0) / (last.qEnd - last.qBegin);
    for (std::size_t j = 0; j < profile.size(); ++j) {
        const auto& segment = profile[j];
        const auto i = segment.gridIndex;
        const double slope0 = i == 0 ? state.betaCollisionSlope - state.betaMeanSlope
            : (residual[i + 1] - residual[i - 1]) / (2.0 * step);
        const double slope1 = (residual[i + 2] - residual[i]) / (2.0 * step);
        const auto stochastic = hermitePolynomial(residual[i], slope0,
            residual[i + 1], slope1, step);
        const double width = segment.qEnd - segment.qBegin;
        const double value0 = j == 0 ? 0.0
            : stochastic.value(segment.fractionBegin) + segment.mean.value(0.0);
        const double value1 = j + 1 == profile.size() ? 0.0
            : stochastic.value(segment.fractionEnd) + segment.mean.value(1.0);
        const double derivative0 = j == 0 ? state.betaCollisionSlope
            : stochastic.derivative(segment.fractionBegin) / step + segment.mean.derivative(0.0) / width;
        const double derivative1 = stochastic.derivative(segment.fractionEnd) / step +
            segment.mean.derivative(1.0) / width;
        const auto total = hermitePolynomial(value0, derivative0, value1, derivative1, width);
        // On a cubic piece, endpoints and derivative roots determine the
        // minimum. Exclude only the two prescribed zero endpoints, never a
        // small distance neighborhood; this rejects hidden early crossings.
        for (double fraction : hermiteKnots(total)) {
            if ((j == 0 && fraction == 0.0) ||
                (j + 1 == profile.size() && fraction == 1.0)) continue;
            const double value = fraction == 0.0 ? value0
                : fraction == 1.0 ? value1 : total.value(fraction);
            if (!(value > 0.0)) {
                sample.survived = false;
                return sample;
            }
        }
    }
    return sample;
}

void runFixedEndpointSampling(const FirstPassageExperimentConfig& config) {
    const int count = config.fixedEndpoint.trajectories > 0
        ? config.fixedEndpoint.trajectories : config.trajectories;
    auto samplesOutput = openOutput(config.outputDirectory / "fixed_endpoint_samples.csv");
    samplesOutput << "kernel_id,kernel_type,state_id,sigma,ell,alpha,beta_0,beta_a,beta_g,"
        "target_distance,target_q,base_step,training_resolution,trajectory,seed,start_value,"
        "endpoint_value,birth_slope_error,endpoint_slope,endpoint_derivative,survived_to_endpoint,"
        "downcrossing,flux_weight_q,flux_weight,mean_profile_type\n";
    auto summary = openOutput(config.outputDirectory / "fixed_endpoint_summary.csv");
    summary << "kernel_id,kernel_type,state_id,sigma,ell,alpha,target_distance,target_q,"
        "base_step,training_resolution,proposals,survived,positive_weight_samples,"
        "survival_acceptance,sum_weight_q,sum_weight_q_squared,effective_samples,"
        "weighted_incoming_slope_mean,weighted_incoming_slope_stddev,"
        "endpoint_slope_mean,endpoint_slope_stddev,endpoint_slope_gaussian_mean,"
        "endpoint_slope_gaussian_stddev,target_field_density,first_passage_density_q,"
        "first_passage_density_distance,max_start_value_error,max_endpoint_value_error,"
        "max_birth_slope_error,status\n";
    const double fineStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    for (std::size_t k = 0; k < config.kernels.size(); ++k) {
        const auto& kernel = config.kernels[k];
        auto normalized = kernel;
        normalized.variance = normalized.lengthScale = 1.0;
        const StationaryKernel covariance(normalized);
        const double sigma = std::sqrt(kernel.variance), derivativeScale = sigma / kernel.lengthScale;
        for (std::size_t s = 0; s < config.collisionStates.size(); ++s) {
            const auto state = resolveCollisionState(config, kernel, config.collisionStates[s]);
            for (std::size_t d = 0; d < config.fixedEndpoint.distances.size(); ++d) {
                const double distance = config.fixedEndpoint.distances[d], targetQ = distance / kernel.lengthScale;
                auto targetConfig = config;
                targetConfig.maximumTime = targetQ;
                for (std::size_t r = 0; r < config.stepSizes.size(); ++r) {
                    const double step = config.stepSizes[r];
                    const auto endpointIndex = static_cast<std::size_t>(std::llround(targetQ / step));
                    const auto profile = prepareMeanProfile(targetConfig, kernel, state, step, true);
                    const double endpointResidual = -profile.back().mean.value(1.0);
                    CirculantSampler diagnostic(covariance, endpointIndex + 2, step, config.maximumEmbeddingExpansions);
                    diagnostic.prepareFixedEndpoint(endpointIndex);
                    const auto gaussian = diagnostic.bridgeSlopeMoments(-state.beta0,
                        state.betaCollisionSlope - state.betaMeanSlope, endpointResidual);
                    const double gaussianMean = gaussian.first + profile.back().mean.derivative(1.0) /
                        (profile.back().qEnd - profile.back().qBegin);
                    const double valueMean = diagnostic.startConditionedMean(endpointIndex,
                        -state.beta0, state.betaCollisionSlope - state.betaMeanSlope);
                    const double valueStddev = std::sqrt(diagnostic.endpointVariance());
                    const double densityQ = normalPdf((endpointResidual - valueMean) / valueStddev) / valueStddev;
                    std::vector<FixedEndpointSample> samples(static_cast<std::size_t>(count));
                    unsigned workersCount = config.threadCount > 0 ? config.threadCount : std::thread::hardware_concurrency();
                    workersCount = std::max(1u, std::min(workersCount, static_cast<unsigned>(count)));
                    std::vector<std::thread> workers;
                    std::vector<std::exception_ptr> errors(workersCount);
                    for (unsigned worker = 0; worker < workersCount; ++worker) {
                        const int begin = static_cast<int>(static_cast<std::uint64_t>(count) * worker / workersCount);
                        const int end = static_cast<int>(static_cast<std::uint64_t>(count) * (worker + 1) / workersCount);
                        workers.emplace_back([&, worker, begin, end]() {
                            try {
                                CirculantSampler sampler(covariance, endpointIndex + 2, step, config.maximumEmbeddingExpansions);
                                sampler.prepareFixedEndpoint(endpointIndex);
                                for (int i = begin; i < end; ++i) {
                                    const auto seed = collisionSeed(config.seed ^ mixSeed(0x656e64706f696e74ULL + d),
                                        k, s, r, static_cast<std::uint64_t>(i));
                                    samples[i] = sampleFixedEndpoint(sampler, state, profile, endpointIndex,
                                        step, endpointResidual, seed);
                                }
                            } catch (...) { errors[worker] = std::current_exception(); }
                        });
                    }
                    for (auto& worker : workers) worker.join();
                    for (const auto& error : errors) if (error) std::rethrow_exception(error);
                    double weight = 0.0, squaredWeight = 0.0, weightedSlope = 0.0, weightedSquare = 0.0;
                    double slopeSum = 0.0, slopeSquareSum = 0.0, maxStart = 0.0, maxEnd = 0.0, maxBirth = 0.0;
                    std::size_t survived = 0, positive = 0;
                    for (std::size_t i = 0; i < samples.size(); ++i) {
                        const auto& item = samples[i];
                        const double w = item.survived ? std::max(0.0, -item.slope) : 0.0;
                        survived += item.survived;
                        positive += w > 0.0;
                        weight += w;
                        squaredWeight += w * w;
                        weightedSlope += w * (-item.slope);
                        weightedSquare += w * item.slope * item.slope;
                        slopeSum += item.slope;
                        slopeSquareSum += item.slope * item.slope;
                        maxStart = std::max(maxStart, std::abs(item.startValue));
                        maxEnd = std::max(maxEnd, std::abs(item.endpointValue));
                        maxBirth = std::max(maxBirth, std::abs(item.birthSlopeError));
                        samplesOutput << csv(kernel.id) << ',' << kernel.type << ',' << csv(state.id) << ','
                            << sigma << ',' << kernel.lengthScale << ',' << kernel.alpha << ','
                            << state.beta0 << ',' << state.betaMeanSlope << ',' << (config.initialConditionType=="positive_exterior" ? std::string{} : preciseString(state.betaCollisionSlope)) << ','
                            << distance << ',' << targetQ << ',' << step << ',' << (step == fineStep ? 1 : 0) << ','
                            << i << ',' << item.seed << ',' << item.startValue * sigma << ',' << item.endpointValue * sigma << ','
                            << item.birthSlopeError << ',' << item.slope << ',' << item.slope * derivativeScale << ','
                            << (item.survived ? 1 : 0) << ',' << (item.slope < 0.0 ? 1 : 0) << ','
                            << w << ',' << w * derivativeScale << ',' << (config.processMeanField ? "full_field" : "affine") << '\n';
                    }
                    const double mean = slopeSum / count;
                    const double weightedMean = weight > 0.0 ? weightedSlope / weight
                        : std::numeric_limits<double>::quiet_NaN();
                    summary << csv(kernel.id) << ',' << kernel.type << ',' << csv(state.id) << ',' << sigma << ','
                        << kernel.lengthScale << ',' << kernel.alpha << ',' << distance << ',' << targetQ << ',' << step << ','
                        << (step == fineStep ? 1 : 0) << ',' << count << ',' << survived << ',' << positive << ','
                        << static_cast<double>(survived) / count << ',' << weight << ',' << squaredWeight << ','
                        << (squaredWeight > 0.0 ? weight * weight / squaredWeight : 0.0) << ',' << weightedMean << ','
                        << (weight > 0.0 ? std::sqrt(std::max(0.0, weightedSquare / weight - weightedMean * weightedMean))
                            : std::numeric_limits<double>::quiet_NaN()) << ',' << mean << ','
                        << std::sqrt(std::max(0.0, slopeSquareSum / count - mean * mean)) << ',' << gaussianMean << ','
                        << gaussian.second << ',' << densityQ / sigma << ',' << densityQ * weight / count << ','
                        << densityQ * weight / count / kernel.lengthScale << ',' << maxStart << ',' << maxEnd << ','
                        << maxBirth << ',' << (weight > 0.0 ? "ok" : "no_positive_weight") << '\n';
                }
            }
        }
    }
}

void runCollisionStateFirstPassage(const FirstPassageExperimentConfig& config) {
    std::optional<std::ofstream> samples;
    if (config.writeRawSamples) {
        samples.emplace(openOutput(config.outputDirectory / "first_passage_samples.csv"));
        *samples << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,"
                    "state_id,beta_0,beta_a,beta_g,max_q,"
                    "base_step,training_resolution,trajectory,event_q,event_distance,event,censored,"
                    "crossing_slope,crossing_derivative,seed,integrator_tolerance,minimum_step,"
                    "bridge_sigma_margin,transitions,bridge_refinements,deepest_refinement";
        if (config.processMeanField) *samples << ",mean_profile_type";
        *samples << '\n';
    }
    std::ofstream summary = openOutput(config.outputDirectory / "first_passage_summary.csv");
    summary << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,"
               "state_id,beta_0,beta_a,beta_g,max_q,base_step,"
               "training_resolution,trajectories,events,censored,event_probability,"
               "restricted_mean_q,event_q10,event_q50,event_q90,crossing_slope_q10,"
               "crossing_slope_q50,crossing_slope_q90,mean_transitions,mean_bridge_refinements";
    if (config.processMeanField) summary << ",mean_profile_type";
    summary << '\n';
    std::ofstream curves = openOutput(config.outputDirectory / "first_passage_curves.csv");
    curves << "kernel_id,kernel_type,sampler_type,crossing_slope_method,sigma,ell,alpha,state_id,"
              "beta_0,beta_a,beta_g,base_step,"
              "training_resolution,bin,q_begin,q_end,at_risk,events,survival,"
              "first_passage_density,hazard_mc,cumulative_hazard_nelson_aalen,"
              "cumulative_hazard_product_limit";
    if (config.processMeanField) curves << ",mean_profile_type";
    curves << '\n';

    std::optional<std::ofstream> profiles;
    if (config.processMeanField) {
        profiles.emplace(openOutput(config.outputDirectory / "first_passage_mean_profiles.csv"));
        *profiles << "kernel_id,state_id,q,distance,mean,beta_mean\n";
    }

    auto meanSegments=openOutput(config.outputDirectory / "first_passage_mean_segments.csv");
    meanSegments << "kernel_id,state_id,start_mode,segment,x_begin,x_end,b0,b1,dx_db0,dx_db1,log_dx,cubic_a,cubic_b,cubic_c,cubic_d\n";
    const double trainingStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    for (std::size_t kernelIndex = 0; kernelIndex < config.kernels.size(); ++kernelIndex) {
        const FirstPassageKernelConfig& kernel = config.kernels[kernelIndex];
        for (std::size_t stateIndex = 0; stateIndex < config.collisionStates.size(); ++stateIndex) {
            FirstPassageCollisionState state = config.collisionStates[stateIndex];
            if (config.processMeanField) {
                const auto& ray = *state.ray;
                const auto jet = config.processMeanField->evaluate(ray.origin);
                const double sigma = std::sqrt(kernel.variance);
                state.beta0 = jet.value / sigma;
                state.betaMeanSlope = rayMeanProfile(config,kernel,state,config.profileMaximumStep)
                    .segments().front().derivative(0.0);
                state.betaCollisionSlope = config.initialConditionType=="positive_exterior" ? 0.0
                    : kernel.lengthScale * ray.gradient.dot(ray.direction) / sigma;
                // Record the actual profile at the finest grid and voxel faces.
                const auto prepared = prepareMeanProfile(config, kernel, state, trainingStep);
                for (std::size_t i = 0; i <= prepared.size(); ++i) {
                    const double q = i == 0 ? 0.0 : prepared[i - 1].qEnd;
                    const double mean = config.processMeanField->evaluate(
                        ray.origin + q * kernel.lengthScale * ray.direction).value;
                    *profiles << csv(kernel.id) << ',' << csv(state.id) << ',' << q << ','
                              << q * kernel.lengthScale << ',' << mean << ',' << mean / sigma << '\n';
                }
            }
            const auto exportedProfile=rayMeanProfile(config,kernel,state,config.profileMaximumStep);
            for (std::size_t i=0; i<exportedProfile.segments().size(); ++i) {
                const auto& segment=exportedProfile.segments()[i];
                meanSegments << csv(kernel.id) << ',' << csv(state.id) << ','
                    << config.initialConditionType << ',' << i << ',' << segment.begin << ',' << segment.end;
                for (double feature : segment.features()) meanSegments << ',' << feature;
                const auto& p=segment.polynomial;
                meanSegments << ',' << p.a << ',' << p.b << ',' << p.c << ',' << p.d << '\n';
            }
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
                writeCollisionCurves(
                    curves, config, kernel, state, step, trainingResolution, generated);
            }
        }
    }
}

} // namespace

void runFirstPassageExperiment(const FirstPassageExperimentConfig& config) {
    // Fail before writing or replacing any experiment output.
    validateFixedEndpoints(config);
    validateMeanFieldRays(config);
    std::filesystem::create_directories(config.outputDirectory);
    writeResolvedFirstPassageConfig(config,
        config.outputDirectory / "resolved_first_passage_config.json");
    if (config.initialConditionType != "fixed_value") {
        if (!config.fixedEndpoint.only) runCollisionStateFirstPassage(config);
        if (config.fixedEndpoint.enabled) runFixedEndpointSampling(config);
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
}

} // namespace mf

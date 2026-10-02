#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/FirstPassageRice.h"

#include "macrofacet/core/Random.h"
#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/FFT>
#include <algorithm>
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

    std::size_t embeddingSize() const { return embeddingSize_; }
    double minimumEigenvalue() const { return minimumEigenvalue_; }

private:
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
        if (type != "fixed_value") {
            throw std::invalid_argument(
                "first_passage.initial_condition currently supports only fixed_value");
        }
        config.initialValue = initial.at("value").get<double>();
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
        const std::string type = sampler.value("type", "exact_grid_circulant");
        if (type != "exact_grid_circulant") {
            throw std::invalid_argument("first_passage.sampler.type must be exact_grid_circulant");
        }
        config.maximumEmbeddingExpansions = sampler.value(
            "max_embedding_expansions", config.maximumEmbeddingExpansions);
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
    requireFinite(config.initialValue, "first_passage.initial_condition.value");
    if (!(config.initialValue > config.threshold)) {
        throw std::invalid_argument("initial_condition.value must be above process.threshold");
    }
    requirePositive(config.maximumTime, "first_passage.grid.max_time");
    if (config.stepSizes.empty()) throw std::invalid_argument("grid.step_sizes cannot be empty");
    for (double step : config.stepSizes) requirePositive(step, "first_passage grid step");
    std::sort(config.stepSizes.begin(), config.stepSizes.end(), std::greater<double>());
    config.stepSizes.erase(std::unique(config.stepSizes.begin(), config.stepSizes.end(),
        [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(a, b); }),
        config.stepSizes.end());
    const double fineStep = *std::min_element(config.stepSizes.begin(), config.stepSizes.end());
    if (!closeInteger(config.maximumTime / fineStep)) {
        throw std::invalid_argument("grid.max_time must be an integer multiple of the finest step");
    }
    for (double step : config.stepSizes) {
        if (!closeInteger(step / fineStep) || !closeInteger(config.maximumTime / step)) {
            throw std::invalid_argument(
                "every grid step must be an integer multiple of the finest step and divide max_time");
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
    Json root{
        {"schema_version", config.schemaVersion},
        {"seed", config.seed},
        {"first_passage", {
            {"process", {{"mean", config.processMean}, {"threshold", config.threshold}}},
            {"initial_condition", {{"type", "fixed_value"}, {"value", config.initialValue}}},
            {"grid", {{"max_time", config.maximumTime}, {"step_sizes", config.stepSizes}}},
            {"curve", {{"bins", config.curveBins}}},
            {"monte_carlo", {
                {"trajectories", config.trajectories},
                {"thread_count", config.threadCount},
                {"confidence_level", config.confidenceLevel},
                {"write_raw_samples", config.writeRawSamples},
                {"minimum_risk_set_for_error", config.minimumRiskSetForError}}},
            {"sampler", {{"type", "exact_grid_circulant"},
                         {"max_embedding_expansions", config.maximumEmbeddingExpansions}}},
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

void runFirstPassageExperiment(const FirstPassageExperimentConfig& config) {
    std::filesystem::create_directories(config.outputDirectory);
    writeResolvedFirstPassageConfig(config,
        config.outputDirectory / "resolved_first_passage_config.json");
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

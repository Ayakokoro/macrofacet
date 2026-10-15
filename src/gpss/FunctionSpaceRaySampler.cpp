#include "macrofacet/gpss/FunctionSpaceRaySampler.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {
namespace {
struct Observation { double q; bool derivative; };

double covariance(const Observation& a, const Observation& b) {
    const double d = a.q - b.q;
    const double rho = std::exp(-0.5*d*d);
    if (a.derivative && b.derivative) return (1.0-d*d)*rho;
    if (a.derivative) return -d*rho;
    if (b.derivative) return d*rho;
    return rho;
}

Eigen::MatrixXd covariance(const std::vector<Observation>& a,
                           const std::vector<Observation>& b) {
    Eigen::MatrixXd result(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j)
            result(i,j) = covariance(a[i],b[j]);
    return result;
}
}

FunctionSpaceRaySampler::FunctionSpaceRaySampler(FunctionSpaceSamplerConfig config)
    : config_(config) {
    if (config.historyPoints < 1 || config.historyPoints > 128 ||
        config.batchSize < 1 || config.batchSize > 128 ||
        !(config.step > 0.0) || !std::isfinite(config.step) ||
        !(config.rankTolerance >= 1e-14 && config.rankTolerance <= 1e-4))
        throw std::invalid_argument("invalid GPIS function-space H, batch size, step or rank tolerance");
    if (config.crossingRule != "tungsten_linear" &&
        config.crossingRule != "hermite_residual_cellwise_mean")
        throw std::invalid_argument("unsupported GPIS crossing_rule");
    plans_.resize(static_cast<std::size_t>(config.historyPoints)+1);
    std::vector<Observation> query;
    for (int j = 1; j <= config.batchSize; ++j) {
        query.push_back({j*config.step,false});
        query.push_back({j*config.step,true});
    }
    const Eigen::MatrixXd queryCovariance = covariance(query,query);
    for (int count = 1;; count = std::min(config.historyPoints,count+config.batchSize)) {
        std::vector<Observation> previous;
        for (int j = 1-count; j <= 0; ++j) previous.push_back({j*config.step,false});
        previous.push_back({0.0,true});
        const Eigen::MatrixXd cross = covariance(query,previous);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance(previous,previous));
        if (eig.info() != Eigen::Success) throw std::runtime_error("GPIS history eigensolver failed");
        const double scale = eig.eigenvalues().maxCoeff();
        if (eig.eigenvalues().minCoeff() < -1e-12*scale)
            throw std::runtime_error("GPIS history covariance is indefinite");
        Eigen::VectorXd inverseRoot = eig.eigenvalues();
        for (Eigen::Index j = 0; j < inverseRoot.size(); ++j)
            inverseRoot[j] = inverseRoot[j] > config.rankTolerance*scale
                ? 1.0/std::sqrt(inverseRoot[j]) : 0.0;
        const Eigen::MatrixXd whitened = cross*eig.eigenvectors()*inverseRoot.asDiagonal();
        auto& plan = plans_[count];
        plan.regression = whitened*inverseRoot.asDiagonal()*eig.eigenvectors().transpose();
        Eigen::MatrixXd conditional = queryCovariance-whitened*whitened.transpose();
        conditional = (0.5*(conditional+conditional.transpose())).eval();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> residual(conditional);
        if (residual.info() != Eigen::Success)
            throw std::runtime_error("GPIS conditional eigensolver failed");
        // Clipping is restricted to numerical roundoff; never add white noise.
        const double tolerance = 4096.0*std::numeric_limits<double>::epsilon()*query.size();
        if (residual.eigenvalues().minCoeff() < -tolerance)
            throw std::runtime_error("GPIS conditional covariance is indefinite; increase rank_tolerance");
        plan.squareRoot = residual.eigenvectors()*
            residual.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
        if (count == config.historyPoints) break;
    }
}

std::vector<FunctionSpaceState> FunctionSpaceRaySampler::next(
    FunctionSpaceHistory& history, Random& rng) const {
    const std::size_t count = history.values.size();
    if (count == 0 || count >= plans_.size() || plans_[count].regression.size() == 0)
        throw std::invalid_argument("invalid GPIS function-space history size");
    Eigen::VectorXd observed(count+1);
    for (std::size_t i = 0; i < count; ++i) observed[i] = history.values[i];
    observed[count] = history.derivative;
    if (!observed.allFinite()) throw std::invalid_argument("nonfinite GPIS history");
    Eigen::VectorXd white(2*config_.batchSize);
    for (Eigen::Index i = 0; i < white.size(); ++i) white[i] = rng.standardNormal();
    const auto& plan = plans_[count];
    const Eigen::VectorXd draw = plan.regression*observed+plan.squareRoot*white;
    if (!draw.allFinite()) throw std::runtime_error("nonfinite GPIS realization");
    std::vector<FunctionSpaceState> result;
    result.reserve(config_.batchSize);
    for (int j = 0; j < config_.batchSize; ++j) {
        result.push_back({draw[2*j],draw[2*j+1]});
        history.values.push_back(draw[2*j]);
    }
    if (history.values.size() > static_cast<std::size_t>(config_.historyPoints))
        history.values.erase(history.values.begin(),history.values.end()-config_.historyPoints);
    history.derivative = result.back().derivative;
    return result;
}

FunctionSpaceFirstPassageSample sampleFunctionSpaceFirstPassage(
    const FunctionSpaceRaySampler& sampler, const RayMeanProfile& mean,
    FunctionSpaceStart start, double outwardDerivative, Random& rng) {
    const double horizon = mean.maximumX(), step = sampler.config().step;
    const bool linear = sampler.config().crossingRule == "tungsten_linear";
    if (mean.segments().empty() || !(horizon > 0.0) ||
        !std::isfinite(horizon/step) || horizon/step > 10000000.0)
        throw std::invalid_argument("invalid GPIS first-passage horizon");
    const auto& first = mean.segments().front();
    const double b0 = first.value(0.0);
    FunctionSpaceState left;
    if (start == FunctionSpaceStart::SurfaceOutward) {
        if (!(outwardDerivative > 0.0) || !std::isfinite(outwardDerivative))
            throw std::invalid_argument("GPIS surface birth requires a positive derivative");
        left = {-b0,outwardDerivative-first.derivative(0.0)};
    } else {
        left.value = start == FunctionSpaceStart::PositiveExterior
            ? std::max(-normalQuantileFromLogCdf(normalLogCdf(b0)+std::log(rng.openUniform01())),
                       std::nextafter(-b0,std::numeric_limits<double>::infinity()))
            : rng.standardNormal();
        left.derivative = rng.standardNormal();
    }
    FunctionSpaceFirstPassageSample result;
    result.distance = horizon;
    result.initialSign = start == FunctionSpaceStart::Unconditioned && b0+left.value < 0 ? -1 : 1;
    FunctionSpaceHistory history{{left.value},left.derivative};
    std::uint64_t gridIndex = 0;
    std::size_t meanIndex = 0;
    double leftTotal = start == FunctionSpaceStart::SurfaceOutward ? 0.0 : b0+left.value;
    while (gridIndex*step < horizon) {
        const auto batch = sampler.next(history,rng);
        result.generatedPoints += batch.size();
        for (const auto& right : batch) {
            const double begin = gridIndex*step;
            const double end = std::min((gridIndex+1)*step,horizon);
            const auto residual = hermitePolynomial(left.value,left.derivative,
                right.value,right.derivative,step);
            if (linear) {
                // Do not insert geometry-cell boundaries into the sampling grid.
                while (meanIndex+1 < mean.segments().size() && mean.segments()[meanIndex].end < end)
                    ++meanIndex;
                // Only a clipped final cell needs residual reconstruction. Never
                // evaluate the mean beyond the valid ray to obtain an endpoint.
                const double endResidual = end == (gridIndex+1)*step ? right.value
                    : residual.value((end-begin)/step);
                const double rightTotal = mean.segments()[meanIndex].value(end)+endResidual;
                const double f0 = result.initialSign*leftTotal, f1 = result.initialSign*rightTotal;
                // A surface birth at q=0 is not a collision. The first chord from
                // zero cannot resolve a positive excursion and an early return.
                if (f0 > 0.0 && f1 <= 0.0) {
                    const double width = end-begin;
                    result.hit = true;
                    result.distance = begin+width*(f0/(f0-f1));
                    result.speed = (f0-f1)/width;
                    return result;
                }
                leftTotal = rightTotal;
                left = right;
                ++gridIndex;
                if (end == horizon) return result;
                continue;
            }
            double lo = begin;
            while (lo < end) {
                while (mean.segments()[meanIndex].end <= lo) ++meanIndex;
                const auto& cell = mean.segments()[meanIndex];
                const double hi = std::min(end,cell.end), width = hi-lo;
                const double u0 = (lo-begin)/step, u1 = (hi-begin)/step;
                auto total = hermitePolynomial(residual.value(u0),residual.derivative(u0)/step,
                    residual.value(u1),residual.derivative(u1)/step,width);
                const auto deterministic = cell.restricted(lo,hi).polynomial;
                total = {total.a+deterministic.a,total.b+deterministic.b,
                         total.c+deterministic.c,total.d+deterministic.d};
                if (begin == 0.0 && lo == 0.0 && start == FunctionSpaceStart::SurfaceOutward) {
                    total.d = 0.0;
                    total.c = width*outwardDerivative;
                }
                const double sign = result.initialSign;
                total = {sign*total.a,sign*total.b,sign*total.c,sign*total.d};
                const auto crossing = firstHermiteDowncrossing(total,width,1e-12*width);
                if (crossing.found) {
                    result.hit = true;
                    result.distance = lo+crossing.fraction*width;
                    result.speed = -crossing.derivative;
                    return result;
                }
                lo = hi;
            }
            left = right;
            ++gridIndex;
            if (end == horizon) return result;
        }
    }
    return result;
}

} // namespace mf

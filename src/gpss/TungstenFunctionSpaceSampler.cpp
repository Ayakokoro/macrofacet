#include "macrofacet/gpss/TungstenFunctionSpaceSampler.h"
// Behavior ported from the local sparse-conv-gpis-tungsten CPU function-space
// medium, GaussianProcess.cpp and Gaussian.cpp. Source hashes and differential
// replay data are retained in tests/data/tungsten_function_space_reference.json.
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {
double lerp(double a, double b, double t) { return a*(1-t)+b*t; }

Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& matrix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(matrix);
    if (eigen.info() != Eigen::Success) throw std::runtime_error("Tungsten GP pseudoinverse failed");
    const double eps = 1e6*DBL_EPSILON*eigen.eigenvalues().cwiseAbs().maxCoeff();
    const Eigen::VectorXd inverse = eigen.eigenvalues().cwiseAbs().cwiseLessOrEqual(eps)
        .select(0., eigen.eigenvalues().cwiseInverse());
    const Eigen::MatrixXd u = eigen.eigenvectors()*inverse.cwiseSqrt().asDiagonal();
    return u*u.transpose();
}
}

void TungstenFunctionSpaceConfig::validate() const {
    if (samplePoints < 2 || samplePoints > 128 || !(stepSize >= 0) || !std::isfinite(stepSize))
        throw std::invalid_argument("Tungsten GPIS requires sample_points=2..128 and finite world step_size>=0");
}

TungstenRandom::TungstenRandom(std::uint64_t seed, std::uint64_t sequence)
    : state_(seed), sequence_(sequence) {
    uniform01(); uniform01(); // UniformSampler constructor calls next2D().
    draws_ = 0;
}
TungstenRandom::TungstenRandom(std::function<double()> source) : source_(std::move(source)) {}

double TungstenRandom::uniform01() {
    ++draws_;
    if (source_) {
        const double value = source_();
        if (!(value >= 0 && value < 1)) throw std::runtime_error("invalid Tungsten replay uniform");
        return value;
    }
    const auto old = state_;
    state_ = old*6364136223846793005ULL+(sequence_|1);
    const auto shifted = static_cast<std::uint32_t>(((old>>18u)^old)>>27u);
    const auto rotation = static_cast<std::uint32_t>(old>>59u);
    const std::uint32_t bits = (shifted>>rotation)|(shifted<<((32u-rotation)&31));
    return static_cast<double>(bits>>9u)*(1.0/8388608.0);
}

Vector2 TungstenRandom::normalPair() {
    const double u1 = uniform01(), u2 = uniform01();
    const double radius = std::sqrt(-2*std::log(1-u1));
    return {radius*std::cos(2*kPi*u2), radius*std::sin(2*kPi*u2)};
}

Eigen::VectorXd TungstenRandom::normals(int count) {
    Eigen::VectorXd result(count);
    for (int i=0; i<count; i+=2) {
        const auto pair = normalPair();
        result[i] = pair.x();
        if (i+1<count) result[i+1] = pair.y(); // odd size discards the second variate
    }
    return result;
}

double TungstenRandom::positiveStart(double mean, double sigma) {
    // Preserve rand_truncated_normal, including its RNG-consuming unused draw
    // in the first batch (numCondPts=0 in FunctionSpaceGaussianProcessMedium).
    if (std::abs(mean) < 0.000001) return std::abs(mean+sigma*normalPair().x());
    if (mean > 0) {
        for (;;) {
            const double value = mean+sigma*normalPair().x();
            if (value >= 0) return value;
        }
    }
    const double a = -mean/sigma;
    double x = 0;
    for (int i=0; i<1000; ++i) {
        x = std::sqrt(a*a-2*std::log(1-uniform01()));
        if (uniform01() < x/a) break;
    }
    return std::max(0.,sigma*x+mean);
}

TungstenFunctionSpaceSampler::TungstenFunctionSpaceSampler(
    const GPSSField& field, TungstenFunctionSpaceConfig config) : field_(field), config_(config) {
    config.validate();
    if (!field.mean || field.kernel.type() != CovarianceKernelType::SquaredExponential)
        throw std::invalid_argument("Tungsten function-space compatibility requires a mean and SE kernel");
    if (Eigen::LLT<Matrix3>(field.kernel.metric()).info() != Eigen::Success)
        throw std::invalid_argument("Tungsten function-space kernel metric must be positive definite");
}

double TungstenFunctionSpaceSampler::covariance(
    const TungstenObservation& a, const TungstenObservation& b) const {
    const auto jet = field_.kernel.evaluate(a.position,b.position);
    if (a.derivative && b.derivative) return a.direction.dot(jet.gradientXGradientY*b.direction);
    if (a.derivative) return a.direction.dot(jet.gradientXValueY);
    if (b.derivative) return b.direction.dot(jet.valueXGradientY);
    return jet.valueValue;
}

Eigen::VectorXd TungstenFunctionSpaceSampler::mean(const std::vector<TungstenObservation>& points) const {
    Eigen::VectorXd result(points.size());
    for (std::size_t i=0; i<points.size(); ++i) {
        const auto jet = field_.mean->evaluateTungsten(points[i].position);
        result[i] = points[i].derivative ? points[i].direction.dot(jet.gradient) : jet.value;
    }
    return result;
}

TungstenGaussian TungstenFunctionSpaceSampler::conditional(
    const std::vector<TungstenObservation>& query, const std::vector<TungstenObservation>& obs,
    const Eigen::VectorXd& values) const {
    if (query.empty() || values.size() != static_cast<Eigen::Index>(obs.size()))
        throw std::invalid_argument("invalid Tungsten GP observations");
    const auto symmetric = [&](const std::vector<TungstenObservation>& points) {
        Eigen::MatrixXd matrix(points.size(),points.size());
        for (std::size_t i=0; i<points.size(); ++i)
            for (std::size_t j=0; j<=i; ++j) matrix(i,j)=matrix(j,i)=covariance(points[i],points[j]);
        return matrix;
    };
    TungstenGaussian result{mean(query),symmetric(query)};
    if (!obs.empty()) {
        const Eigen::MatrixXd s11 = symmetric(obs);
        Eigen::MatrixXd s12(obs.size(),query.size());
        for (std::size_t i=0; i<obs.size(); ++i)
            for (std::size_t j=0; j<query.size(); ++j) s12(i,j)=covariance(obs[i],query[j]);
        const Eigen::MatrixXd solved = (pseudoInverse(s11)*s12).transpose();
        result.mean += solved*(values-mean(obs));
        result.covariance -= solved*s12;
    }
    return result;
}

Eigen::VectorXd TungstenFunctionSpaceSampler::draw(const TungstenGaussian& gaussian, TungstenRandom& rng) {
    Eigen::LLT<Eigen::MatrixXd> chol(gaussian.covariance.triangularView<Eigen::Lower>());
    Eigen::MatrixXd transform;
    if (chol.info() == Eigen::Success) transform = chol.matrixL();
    else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(gaussian.covariance);
        if (eigen.info() != Eigen::Success) throw std::runtime_error("Tungsten GP covariance decomposition failed");
        transform = eigen.eigenvectors()*eigen.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    }
    Eigen::VectorXd result = gaussian.mean+transform*rng.normals(static_cast<int>(gaussian.mean.size()));
    if (!result.allFinite()) throw std::runtime_error("nonfinite Tungsten GP realization");
    return result;
}

void TungstenFunctionSpaceSampler::locateIntersection(TungstenFunctionSpaceBatch& batch,
    const Ray& ray, bool skipFirstInterval) {
    const Vector3 direction = ray.direction.normalized();
    double previous = batch.values[0], previousT = batch.times[0];
    int sign = previous < 0 ? -1 : 1;
    batch.initialSign = sign;
    for (int p=1; p<batch.values.size(); ++p) {
        const double current = batch.values[p], currentT = batch.times[p];
        const int currentSign = current < 0 ? -1 : 1;
        if (skipFirstInterval && p==1) sign = currentSign;
        else if (currentSign != sign) {
            const double fraction = previous/(previous-current);
            batch.distance = lerp(previousT,currentT,fraction);
            batch.values.conservativeResize(p+2);
            batch.values[p] = lerp(previous,current,fraction);
            batch.values[p+1] = (previous-current)/(previousT-currentT);
            batch.observations.resize(p+2);
            const Point3 point = ray.origin+batch.distance*direction;
            batch.observations[p] = {point,false,Vector3::Zero()};
            // sampleGrad uses the original (float) ray direction as deriv_dir.
            batch.observations[p+1] = {point,true,ray.direction};
            batch.hit = true;
            break;
        }
        previous=current; previousT=currentT;
    }
    batch.end.position = ray.origin+batch.distance*direction;
    batch.end.value = batch.values[batch.values.size()-(batch.hit ? 2 : 1)];
}

Vector3 TungstenFunctionSpaceSampler::sampleGradient(const TungstenFunctionSpaceBatch& batch,
    const Vector3& direction, TungstenRandom& rng) const {
    // TangentFrameD, including its use of the unnormalized float ray direction.
    const double sign = direction.z()<0 ? -1. : 1.;
    const double a = -1./(sign+direction.z()), b = direction.x()*direction.y()*a;
    const Vector3 tangent(1+sign*direction.x()*direction.x()*a,sign*b,-sign*direction.x());
    const Vector3 bitangent(b,sign+direction.y()*direction.y()*a,-direction.y());
    std::vector<TungstenObservation> query{{batch.end.position,true,tangent},{batch.end.position,true,bitangent}};
    if (!batch.hit) query.push_back({batch.end.position,true,direction});
    const auto g = draw(conditional(query,batch.observations,batch.values),rng);
    return tangent*g[0]+bitangent*g[1]+direction*(batch.hit ? batch.values[batch.values.size()-1] : g[2]);
}

TungstenFunctionSpaceBatch TungstenFunctionSpaceSampler::sampleBatch(const Ray& ray, float nearT,
    float farT, const std::optional<TungstenSurfaceState>& previous, bool firstIntersectionAlongRay,
    TungstenRandom& rng) const {
    // A rounded continuation may land exactly at farT. The upstream loop
    // still samples that final zero-length batch before reporting escape.
    if (!(farT>=nearT) || !std::isfinite(farT)) throw std::invalid_argument("invalid Tungsten GP ray interval");
    const int count = config_.samplePoints;
    const double offset = rng.uniform01();
    const Vector3 direction = ray.direction.normalized();
    double step = (double(farT)-nearT)/count;
    if (config_.stepSize>0 && config_.stepSize<step) step=config_.stepSize;
    const double length = step*count;
    TungstenFunctionSpaceBatch batch;
    batch.distance = nearT+length;
    for (int i=0; i<count; ++i) {
        double t = lerp(nearT+step*0.1,nearT+length,std::clamp((i-offset)/(count-1),0.,1.));
        if (i==0) t=nearT+step*0.1;
        else if (i==count-1) t=nearT+length;
        batch.times.push_back(t);
        batch.observations.push_back({ray.origin+t*direction,false,Vector3::Zero()});
    }
    std::vector<TungstenObservation> obs;
    Eigen::VectorXd values;
    if (previous) {
        obs = {{previous->position,false,Vector3::Zero()},{previous->position,true,direction}};
        values.resize(2); values << previous->value,previous->gradient.dot(direction);
    } else {
        const Point3 start = (ray.origin.cast<float>()+nearT*ray.direction.cast<float>()).cast<double>();
        rng.positiveStart(field_.mean->evaluateTungsten(start).value,field_.kernel.sigma());
    }
    batch.values = draw(conditional(batch.observations,obs,values),rng);
    locateIntersection(batch,ray,previous.has_value() && firstIntersectionAlongRay);
    // Original code also draws a gradient on the final escaped batch.
    batch.end.gradient = sampleGradient(batch,ray.direction,rng);
    return batch;
}
} // namespace mf

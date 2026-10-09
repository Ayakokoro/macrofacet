#include "macrofacet/transport/RenewalMedium.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {

Vector3 sampleRenewalGradient(const CovarianceKernel& kernel, const Vector3& direction,
    double x, double derivative, const Vector3& meanGradient,
    const Vector3& birthMeanGradient, const std::optional<Vector3>& birthGradient, Random& rng) {
    if (kernel.type() != CovarianceKernelType::Matern32 || !(x >= 0) || !std::isfinite(x) ||
        !(derivative < 0) || !std::isfinite(derivative) || !meanGradient.allFinite() ||
        !birthMeanGradient.allFinite() || (birthGradient && !birthGradient->allFinite()))
        throw std::invalid_argument("invalid Renewal gradient reconstruction inputs");
    const Vector3 w = normalizedOrThrow(direction);
    Vector3 e1, e2;
    orthonormalComplement(w, e1, e2);
    Eigen::Matrix<double,3,2> basis;
    basis << e1, e2;
    const Matrix3 C = kernel.gradientCovarianceAtZero();
    const Vector3 cw = C*w;
    const double variance = w.dot(cw);
    if (!(variance > 0)) throw std::invalid_argument("degenerate Renewal directional gradient covariance");
    const Vector3 regression = cw/variance;
    const Matrix3 residualCovariance = C-cw*regression.transpose();
    const Matrix2 transverse = basis.transpose()*residualCovariance*basis;
    const Eigen::LLT<Matrix2> cholesky(transverse);
    if (cholesky.info() != Eigen::Success)
        throw std::invalid_argument("Renewal requires a positive-definite kernel metric");
    Vector2 residual = Vector2::Zero();
    double varianceFactor = 1;
    if (birthGradient) {
        const Vector3 previous = *birthGradient-birthMeanGradient;
        residual = std::exp(-x)*(basis.transpose()*(previous-regression*w.dot(previous)));
        varianceFactor = -std::expm1(-2*x);
    }
    // Explicit temporaries fix random draw order across compilers.
    const double z1 = rng.standardNormal(), z2 = rng.standardNormal();
    const Vector2 noise = cholesky.matrixL()*Vector2(z1,z2);
    residual += std::sqrt(varianceFactor)*noise;
    const Vector3 longitudinal = meanGradient+regression*(derivative-w.dot(meanGradient));
    // Construct in the ray frame so transverse roundoff cannot flip V's sign.
    return derivative*w + basis*(basis.transpose()*longitudinal+residual);
}

RenewalMedium::RenewalMedium(const GPSSField& field, const RenewalHazardModel& model, double maximumStep,
                           const std::string& profileMode)
    : field_(field), model_(model), maximumStep_(maximumStep), pointLinear_(profileMode == "point_linear") {
    if (profileMode != "cubic" && !pointLinear_)
        throw std::invalid_argument("unknown Renewal profile mode: " + profileMode);
    field_.validate();
    if (field_.kernel.type() != CovarianceKernelType::Matern32 || !model_.hasMixture())
        throw std::invalid_argument("neural_renewal requires unit-decay matern_3_2 and a full model export");
    if (!(maximumStep > 0) || !std::isfinite(maximumStep))
        throw std::invalid_argument("invalid Renewal profile maximum step");
    const Eigen::LLT<Matrix3> cholesky(field_.kernel.metric());
    if (cholesky.info() != Eigen::Success)
        throw std::invalid_argument("neural_renewal requires a positive-definite kernel metric");
}

std::optional<RenewalFlight> RenewalMedium::beginFlight(const Ray& input,
    std::optional<Vector3> birthGradient) const {
    Ray ray{input.origin, normalizedOrThrow(input.direction)};
    if (!field_.activeDomain.contains(ray.origin))
        throw std::invalid_argument("Renewal flight origin must be inside the active domain");
    const auto interval = field_.activeDomain.intersect(ray);
    if (!interval.hit || !(interval.exit > 0)) return std::nullopt;
    const double ell = 1/std::sqrt(ray.direction.dot(field_.kernel.metric()*ray.direction));
    const double sigma = field_.kernel.sigma();
    RayStartCondition start;
    if (birthGradient) {
        start = {RayStartMode::SurfaceOutward, ell/sigma*ray.direction.dot(*birthGradient)};
        if (!birthGradient->allFinite() || !(start.outwardDerivative > 0))
            throw std::invalid_argument("Renewal surface flight must point outward with a finite full gradient");
    }
    auto mean = pointLinear_
        ? RayMeanProfile::pointLinear(field_.mean,ray.origin,ray.direction,sigma,ell,interval.exit,maximumStep_)
        : RayMeanProfile::fromField(*field_.mean, ray.origin, ray.direction,
            sigma, ell, interval.exit, maximumStep_);
    const auto& first = mean.segment(0);
    const Point3 interior = ray.origin+(0.5*first.end*ell)*ray.direction;
    const Vector3 birthMean = mean.initialPointJet() ? mean.initialPointJet()->gradient
        : field_.mean->evaluateInCell(ray.origin, interior).gradient;
    return RenewalFlight{ray, std::move(birthGradient), std::move(mean), start, ell, sigma, birthMean};
}

RenewalSurfaceSample RenewalMedium::sample(const RenewalFlight& flight, Random& rng,
    double tolerance, RenewalTrackingDiagnostics* diagnostics) const {
    if (!(tolerance > 0) || !std::isfinite(tolerance)) throw std::invalid_argument("invalid Renewal sample tolerance");
    if (diagnostics) ++diagnostics->flights;
    auto state = model_.initialize(flight.mean.segment(0), flight.start);
    double remaining = -std::log(rng.openUniform01());
    for (std::size_t i = 0; flight.mean.hasSegment(i); ++i) {
        const auto& segment = flight.mean.segment(i);
        auto step = model_.evaluate(state, segment);
        if (diagnostics) ++diagnostics->segments;
        const RenewalHazardSegment hazard(segment.begin, segment.end, step.rates);
        const double mass = hazard.integral(0,1);
        if (mass > 0 && remaining <= mass) {
            const double u = hazard.inverse(remaining, 0, 1, tolerance/flight.ell);
            if (diagnostics) ++diagnostics->mixtureQueries;
            return sampleHit(flight,i,u,model_.mixture(state,segment,u),rng);
        }
        remaining -= mass;
        state = std::move(step.nextState);
    }
    RenewalSurfaceSample miss;
    miss.distance = flight.maximumDistance();
    miss.segment = flight.mean.constructedSegmentCount();
    miss.position = flight.ray.origin+miss.distance*flight.ray.direction;
    return miss;
}

RenewalSurfaceSample RenewalMedium::sampleHit(const RenewalFlight& flight, std::size_t index,
    double u, const RenewalSpeedMixture& mixture, Random& rng) const {
    if (!flight.mean.hasSegment(index) || !(u >= 0 && u <= 1))
        throw std::invalid_argument("invalid Renewal collision coordinate");
    const auto& segment = flight.mean.segment(index);
    const double x = segment.begin+(segment.end-segment.begin)*u;
    RenewalSurfaceSample result;
    result.hit = true; result.segment = index; result.distance = x*flight.ell;
    result.position = flight.ray.origin+result.distance*flight.ray.direction;
    result.position = result.position.cwiseMax(field_.activeDomain.minimum).cwiseMin(field_.activeDomain.maximum);
    result.speed = mixture.sample(rng);
    const Point3 interior = flight.ray.origin+(0.5*(segment.begin+segment.end)*flight.ell)*flight.ray.direction;
    const Vector3 meanGradient = field_.mean->evaluateInCell(result.position,interior).gradient;
    result.gradient = sampleRenewalGradient(field_.kernel,flight.ray.direction,x,
        -flight.sigma/flight.ell*result.speed,meanGradient,flight.birthMeanGradient,flight.birthGradient,rng);
    result.normal = normalizedOrThrow(result.gradient);
    if (!(flight.ray.direction.dot(result.gradient) < 0))
        throw std::runtime_error("Renewal collision lost its inward directional derivative");
    return result;
}

double RenewalMedium::transmittance(const RenewalFlight& flight, double distance) const {
    if (!(distance >= 0) || !std::isfinite(distance)) throw std::invalid_argument("invalid Renewal visibility distance");
    return renewalTransmittance(model_, flight.mean, flight.start, flight.ell, 0,
        std::min(distance, flight.maximumDistance()));
}

double RenewalMedium::surfaceTransmittance(const Ray& ray, const Vector3& gradient, double distance) const {
    if (!(distance >= 0) || !std::isfinite(distance) || !gradient.allFinite())
        throw std::invalid_argument("invalid Renewal surface visibility query");
    if (distance == 0) return 1;
    if (normalizedOrThrow(ray.direction).dot(gradient) <= 0) return 0;
    const auto flight = beginFlight(ray, gradient);
    return flight ? transmittance(*flight, distance) : 1;
}

} // namespace mf

#include "macrofacet/transport/FirstHitSEMedium.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {
class NegatedMean final : public MeanField {
    MeanFieldPtr source_;
    static MeanJet negate(MeanJet j) { j.value = -j.value; j.gradient = -j.gradient; return j; }
public:
    explicit NegatedMean(MeanFieldPtr source) : source_(std::move(source)) {}
    const char* typeName() const override { return "first_hit_negative_side"; }
    std::optional<double> intrinsicSigma() const override { return source_->intrinsicSigma(); }
    MeanJet evaluate(const Point3& x) const override { return negate(source_->evaluate(x)); }
    MeanJet evaluateInCell(const Point3& x, const Point3& cell) const override {
        return negate(source_->evaluateInCell(x, cell));
    }
    MeanRayPoint queryRayPoint(const Point3& o, const Vector3& w, double begin, double end) const override {
        auto p = source_->queryRayPoint(o, w, begin, end); p.jet = negate(p.jet); return p;
    }
    double valueDifference(const Point3& x, const Vector3& d) const override {
        return -source_->valueDifference(x, d);
    }
    BoundsSummary bounds(const Bounds3& domain) const override {
        const auto b = source_->bounds(domain);
        const Point3 center = 0.5*(domain.minimum+domain.maximum);
        const double radius = 0.5*(domain.maximum-domain.minimum).norm();
        return {-source_->evaluate(center).value-b.maximumGradientNorm*radius,
                b.maximumGradientNorm, b.certified};
    }
    MeanRayBounds rayBounds(const Point3& o, const Vector3& w, double begin, double end) const override {
        auto b = source_->rayBounds(o,w,begin,end);
        const double lo = b.minimumValue, dlo = b.minimumDerivative;
        b.minimumValue = -b.maximumValue; b.maximumValue = -lo;
        b.minimumDerivative = -b.maximumDerivative; b.maximumDerivative = -dlo;
        b.beginDerivative = -b.beginDerivative; return b;
    }
    std::optional<Vector3> affineGradient() const override {
        const auto g = source_->affineGradient();
        return g ? std::optional<Vector3>(-*g) : std::nullopt;
    }
    double voxelSizeHint() const override { return source_->voxelSizeHint(); }
    void appendRayBreakpoints(const Point3& o, const Vector3& w, double begin, double end,
        std::vector<double>& knots) const override { source_->appendRayBreakpoints(o,w,begin,end,knots); }
    void requireFullRayCoverage(const Point3& o, const Vector3& w, double end) const override {
        source_->requireFullRayCoverage(o,w,end);
    }
};

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x = (x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
}

std::uint64_t firstHitStreamSeed(std::uint64_t seed, std::uint64_t pixel,
    std::uint64_t sample, std::uint64_t stream) {
    return mix(mix(mix(seed)^pixel)^mix(sample)^mix(stream+0x6669727374686974ULL));
}

GPSSField firstHitSideField(const GPSSField& field, int side) {
    if (side != 1 && side != -1) throw std::invalid_argument("first-hit side must be +1 or -1");
    auto result = field;
    if (side < 0) result.mean = std::make_shared<NegatedMean>(field.mean);
    return result;
}

FirstHitSEMedium::FirstHitSEMedium(const GPSSField& field, std::shared_ptr<const FirstHitSEModel> model,
    double maximumStep, FunctionSpaceSamplerConfig reference)
    : field_(field), model_(std::move(model)), profileMaximumStep_(maximumStep), reference_(reference) {
    field_.validate();
    if (field_.kernel.type() != CovarianceKernelType::SquaredExponential)
        throw std::invalid_argument("first-hit rendering requires a stationary SE kernel");
    if (Eigen::LLT<Matrix3>(field_.kernel.metric()).info() != Eigen::Success)
        throw std::invalid_argument("first-hit rendering requires a positive-definite kernel metric");
    if (!(maximumStep > 0) || !std::isfinite(maximumStep))
        throw std::invalid_argument("invalid first-hit profile maximum step");
}

std::optional<FirstHitSEFlight> FirstHitSEMedium::beginFlight(const Ray& cameraRay) const {
    Ray ray{cameraRay.origin, normalizedOrThrow(cameraRay.direction)};
    const auto domain = field_.activeDomain.intersect(ray);
    if (!domain.hit || !(domain.exit > domain.entry)) return std::nullopt;
    ray.origin += domain.entry*ray.direction;
    ray.origin = ray.origin.cwiseMax(field_.activeDomain.minimum).cwiseMin(field_.activeDomain.maximum);
    const double ell = 1/std::sqrt(ray.direction.dot(field_.kernel.metric()*ray.direction));
    const double sigma = field_.kernel.sigma();
    // Re-clip after moving/clamping the origin. Subtracting the two distant
    // camera intersection parameters can overshoot a very short grazing chord
    // by several ulps, even after the relative endpoint backoff. Coverage must
    // be bounded by the ray we actually pass to the field interpolator.
    const auto remaining = field_.activeDomain.intersect(ray);
    if (!remaining.hit || !(remaining.exit > 0)) return std::nullopt;
    const double maximum = std::min(domain.exit-domain.entry,remaining.exit)*(1-1e-12);
    auto mean = RayMeanProfile::fromField(*field_.mean,ray.origin,ray.direction,
        sigma,ell,maximum,profileMaximumStep_);
    return FirstHitSEFlight{ray,domain.entry,ell,sigma,std::move(mean)};
}

FirstHitSESurfaceSample FirstHitSEMedium::sample(const FirstHitSEFlight& flight,
    Random& eventRng, Random& gradientRng, double tolerance) const {
    if (!(tolerance > 0) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid first-hit physical distance tolerance");
    FirstHitSESample draw;
    if (model_) draw = model_->sample(flight.mean,eventRng,0,tolerance/flight.ell);
    else {
        const auto reference = sampleFunctionSpaceFirstPassage(reference_,flight.mean,
            FunctionSpaceStart::Unconditioned,0,eventRng);
        draw = {reference.hit,reference.distance,reference.speed,reference.initialSign};
    }
    return reconstruct(flight,draw,gradientRng);
}

FirstHitSESurfaceSample FirstHitSEMedium::reconstruct(const FirstHitSEFlight& flight,
    const FirstHitSESample& draw, Random& gradientRng) const {
    if (!(draw.q>=0 && draw.q<=flight.mean.maximumX()) || (draw.initialSign!=1 && draw.initialSign!=-1) ||
        (draw.hit && (!(draw.speed>0) || !std::isfinite(draw.speed))))
        throw std::invalid_argument("invalid first-hit reconstruction sample");
    FirstHitSESurfaceSample result;
    result.enteredDomain = true; result.hit = draw.hit; result.initialSign = draw.initialSign;
    result.entryDistance = flight.entryDistance; result.entry = flight.ray.origin;
    result.maximumDistance = flight.mean.maximumX()*flight.ell;
    result.q = draw.q; result.distance = draw.q*flight.ell; result.speed = draw.speed;
    result.position = flight.ray.origin+result.distance*flight.ray.direction;
    result.observationPosition = result.position;
    if (!draw.hit) return result;
    const auto& segments = flight.mean.segments();
    auto it = std::lower_bound(segments.begin(),segments.end(),draw.q,
        [](const RayMeanSegment& segment,double q) { return segment.end < q; });
    if (it == segments.end()) throw std::runtime_error("first-hit sample outside profile");
    const Point3 interior = flight.ray.origin+0.5*(it->begin+it->end)*flight.ell*flight.ray.direction;
    const Vector3 mu = field_.mean->evaluateInCell(result.position,interior).gradient;
    const double derivative = -draw.initialSign*flight.sigma/flight.ell*draw.speed;
    result.gradient = sampleFirstHitSEGradient(field_.kernel,flight.ray.direction,derivative,mu,gradientRng);
    result.normal = normalizedOrThrow(result.gradient);
    if (!(draw.initialSign*flight.ray.direction.dot(result.gradient) < 0))
        throw std::runtime_error("first-hit gradient lost its crossing sign");
    return result;
}
} // namespace mf

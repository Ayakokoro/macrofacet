#include "macrofacet/gpss/RayMeanProfile.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <utility>

namespace mf {
namespace {
void positive(double x, const char* name) {
    if (!(x>0.0) || !std::isfinite(x)) throw std::invalid_argument(name);
}

std::vector<double> partition(std::vector<double> knots, double maximumStep) {
    positive(maximumStep, "invalid profile maximum step");
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    std::vector<double> result{knots.front()};
    for (std::size_t i=1; i<knots.size(); ++i) {
        const double width=knots[i]-knots[i-1];
        const double requested=std::ceil(width/maximumStep);
        if (!(requested>=1.0) || requested>10000000.0)
            throw std::invalid_argument("ray mean profile has too many segments");
        const auto count=static_cast<std::size_t>(requested);
        for (std::size_t j=1; j<count; ++j)
            result.push_back(knots[i-1]+width*(static_cast<double>(j)/count));
        result.push_back(knots[i]);
    }
    return result;
}
}

double RayMeanSegment::value(double x) const {
    return polynomial.value((x-begin)/(end-begin));
}
double RayMeanSegment::derivative(double x) const {
    return polynomial.derivative((x-begin)/(end-begin))/(end-begin);
}
RayMeanSegment RayMeanSegment::restricted(double lo, double hi) const {
    if (!(lo>=begin && hi<=end && hi>lo))
        throw std::invalid_argument("invalid mean segment restriction");
    return {lo,hi,hermitePolynomial(value(lo),derivative(lo),value(hi),derivative(hi),hi-lo)};
}
std::array<double,5> RayMeanSegment::features() const {
    return {polynomial.value(0.0),polynomial.value(1.0),
            polynomial.derivative(0.0),polynomial.derivative(1.0),std::log(end-begin)};
}

RayMeanProfile::RayMeanProfile(std::vector<RayMeanSegment> segments)
    : segments_(std::move(segments)) {
    if (segments_.empty()) throw std::invalid_argument("empty ray mean profile");
    double previous=0.0;
    for (const auto& segment : segments_) {
        if (segment.begin!=previous || !(segment.end>segment.begin) ||
            !std::isfinite(segment.end) || !std::isfinite(segment.polynomial.a) ||
            !std::isfinite(segment.polynomial.b) || !std::isfinite(segment.polynomial.c) ||
            !std::isfinite(segment.polynomial.d))
            throw std::invalid_argument("invalid or non-contiguous ray mean profile");
        previous=segment.end;
    }
    maximumX_ = previous;
}

struct RayMeanProfile::PointSource {
    std::shared_ptr<const MeanField> mean;
    Point3 origin;
    Vector3 direction;
    double sigma, ell, maximumDistance, maximumStep;
};

RayMeanProfile RayMeanProfile::pointLinear(std::shared_ptr<const MeanField> mean,
    const Point3& origin, const Vector3& direction, double sigma, double ell,
    double maximumDistance, double maximumStep) {
    positive(sigma,"invalid point profile sigma");
    positive(ell,"invalid point profile ell");
    positive(maximumDistance,"invalid point profile distance");
    positive(maximumStep,"invalid point profile step");
    if (!mean || !origin.allFinite() || !direction.allFinite() || std::abs(direction.norm()-1)>1e-10)
        throw std::invalid_argument("point profile requires a field and a finite unit ray");
    RayMeanProfile result;
    result.maximumX_ = maximumDistance/ell;
    positive(result.maximumX_,"invalid normalized point profile distance");
    result.pointSource_ = std::make_shared<PointSource>(PointSource{
        std::move(mean),origin,direction,sigma,ell,maximumDistance,maximumStep});
    result.hasSegment(0);
    return result;
}

bool RayMeanProfile::hasSegment(std::size_t index) const {
    while (segments_.size() <= index && pointSource_ &&
           (segments_.empty() || segments_.back().end < maximumX_)) {
        if (segments_.size() >= 10000000) throw std::runtime_error("too many point-query segments");
        const auto& s = *pointSource_;
        const double lo = segments_.empty() ? 0 : segments_.back().end;
        const double limit = std::min(maximumX_,lo+s.maximumStep);
        const double begin = lo*s.ell;
        const double maximumEnd = limit == maximumX_ ? s.maximumDistance : limit*s.ell;
        if (!(maximumEnd > begin)) throw std::runtime_error("point-query step cannot advance");
        const auto point = s.mean->queryRayPoint(s.origin,s.direction,begin,maximumEnd);
        if (!(point.end > begin) || point.end > maximumEnd || !std::isfinite(point.end) ||
            !std::isfinite(point.jet.value) || !point.jet.gradient.allFinite())
            throw std::runtime_error("invalid field point-query result");
        const double hi = point.end == maximumEnd ? limit : point.end/s.ell;
        if (!(hi > lo)) {
            std::ostringstream message;
            message << std::setprecision(17) << "normalized point-query step cannot advance: lo=" << lo
                << " begin=" << begin << " end=" << point.end << " ell=" << s.ell
                << " origin=" << s.origin.transpose() << " direction=" << s.direction.transpose();
            throw std::runtime_error(message.str());
        }
        const double b = point.jet.value/s.sigma;
        const double delta = (hi-lo)*s.ell/s.sigma*point.jet.gradient.dot(s.direction);
        if (!std::isfinite(b) || !std::isfinite(delta)) throw std::runtime_error("nonfinite point features");
        if (segments_.empty()) initialPointJet_ = point.jet;
        // No cubic recovery: a=b=0, c=the first-order increment, d=current value.
        segments_.push_back({lo,hi,{0,0,delta,b}});
    }
    return index < segments_.size();
}

const RayMeanSegment& RayMeanProfile::segment(std::size_t index) const {
    if (!hasSegment(index)) throw std::out_of_range("ray mean segment index");
    return segments_[index];
}

const std::vector<RayMeanSegment>& RayMeanProfile::segments() const {
    while (hasSegment(segments_.size())) {}
    return segments_;
}

RayMeanProfile RayMeanProfile::affine(double value, double slope, double maximumX,
                                      double maximumStep) {
    positive(maximumX,"invalid profile maximum distance");
    return RayMeanProfile({{0.0,maximumX,{0.0,0.0,slope*maximumX,value}}})
        .subdivided(maximumStep);
}

RayMeanProfile RayMeanProfile::fromField(const MeanField& mean, const Point3& origin,
    const Vector3& direction, double sigma, double ell, double maximumDistance,
    double maximumStep) {
    positive(sigma,"invalid profile sigma");
    positive(ell,"invalid profile ell");
    positive(maximumDistance,"invalid profile physical distance");
    if (!origin.allFinite() || !direction.allFinite() ||
        std::abs(direction.norm()-1.0)>1e-10)
        throw std::invalid_argument("profile requires finite origin and unit direction");
    mean.requireFullRayCoverage(origin,direction,maximumDistance);
    std::vector<double> knots{0.0,maximumDistance};
    mean.appendRayBreakpoints(origin,direction,0.0,maximumDistance,knots);
    for (double& t : knots) t/=ell;
    knots=partition(std::move(knots),maximumStep);
    std::vector<RayMeanSegment> segments;
    const auto sample=[&](double x) { return mean.evaluate(origin+(ell*x)*direction).value/sigma; };
    for (std::size_t i=1; i<knots.size(); ++i) {
        const double lo=knots[i-1], hi=knots[i], width=hi-lo;
        const double v0=sample(lo), v1=sample(lo+width/3.0);
        const double v2=sample(lo+2.0*width/3.0), v3=sample(hi);
        // Four values recover the cubic without querying ambiguous gradients
        // on a voxel face. Both endpoint derivatives belong to this cell.
        segments.push_back({lo,hi,{
            (-9*v0+27*v1-27*v2+9*v3)/2,
            (18*v0-45*v1+36*v2-9*v3)/2,
            (-11*v0+18*v1-9*v2+2*v3)/2,v0}});
    }
    return RayMeanProfile(std::move(segments));
}

RayMeanProfile RayMeanProfile::subdivided(double maximumStep) const {
    positive(maximumStep,"invalid profile subdivision step");
    std::vector<RayMeanSegment> result;
    for (const auto& segment : segments()) {
        const auto knots=partition({segment.begin,segment.end},maximumStep);
        for (std::size_t i=1; i<knots.size(); ++i)
            result.push_back(segment.restricted(knots[i-1],knots[i]));
    }
    return RayMeanProfile(std::move(result));
}

} // namespace mf

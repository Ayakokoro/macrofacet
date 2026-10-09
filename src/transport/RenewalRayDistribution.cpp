#include "macrofacet/transport/RenewalRayDistribution.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {
void unit(double u) {
    if (!(u >= 0 && u <= 1)) throw std::invalid_argument("Renewal local coordinate must be in [0,1]");
}
void interval(double from, double to, double maximum, double ell) {
    if (!(ell > 0) || !std::isfinite(ell) || !std::isfinite(maximum) ||
        !(from >= 0 && to >= from && to <= maximum))
        throw std::invalid_argument("invalid physical Renewal query interval or length scale");
}
void sampleArguments(double depth, double tolerance) {
    if (!(depth >= 0) || !(tolerance > 0) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid Renewal optical depth or inversion tolerance");
}
template<std::size_t N>
double deCasteljau(std::array<double, N> values, double u) {
    for (std::size_t n = N-1; n > 0; --n)
        for (std::size_t j = 0; j < n; ++j) values[j] = (1-u)*values[j] + u*values[j+1];
    return values[0];
}
double local(const RenewalHazardSegment& segment, double x) {
    return std::clamp((x-segment.begin())/(segment.end()-segment.begin()), 0.0, 1.0);
}
bool consume(const RenewalHazardSegment& segment, double fromX, double toX, double ell,
    double tolerance, double& remaining, double& distance) {
    const double u0 = local(segment, fromX), u1 = local(segment, toX);
    const double mass = segment.integral(u0, u1);
    if (mass > 0 && remaining <= mass) {
        const double u = segment.inverse(remaining, u0, u1, tolerance/ell);
        distance = (segment.begin()+(segment.end()-segment.begin())*u)*ell;
        return true;
    }
    remaining -= mass;
    return false;
}
} // namespace

RenewalHazardSegment::RenewalHazardSegment(double begin, double end, std::array<double, 4> rates)
    : begin_(begin), end_(end), rates_(rates) {
    if (!(begin >= 0 && end > begin) || !std::isfinite(end))
        throw std::invalid_argument("invalid Renewal hazard segment bounds");
    for (double r : rates) if (!(r >= 0) || !std::isfinite(r))
        throw std::invalid_argument("Renewal hazard coefficients must be finite and nonnegative");
    if (!std::isfinite(cumulative(1))) throw std::invalid_argument("Renewal segment optical depth overflow");
}
double RenewalHazardSegment::hazard(double u) const {
    unit(u);
    return deCasteljau(rates_, u);
}
double RenewalHazardSegment::cumulative(double u) const {
    unit(u);
    std::array<double, 5> coefficients{};
    for (int j = 0; j < 4; ++j)
        coefficients[j+1] = coefficients[j]+((end_-begin_)/4)*rates_[j];
    return deCasteljau(coefficients, u);
}
double RenewalHazardSegment::integral(double u0, double u1) const {
    unit(u0); unit(u1);
    if (u1 < u0) throw std::invalid_argument("reversed Renewal local interval");
    // Two-point Gauss quadrature integrates the cubic exactly. Positive terms
    // avoid subtracting large cumulative values for a short surviving interval.
    const double half = 0.5*(u1-u0), middle = 0.5*u0+0.5*u1;
    const double shift = half/std::sqrt(3.0);
    return ((end_-begin_)*(u1-u0))*(0.5*hazard(middle-shift)+0.5*hazard(middle+shift));
}
double RenewalHazardSegment::inverse(double depth, double u0, double u1, double tolerance) const {
    sampleArguments(depth, tolerance);
    const double total = integral(u0, u1);
    if (depth > total) throw std::invalid_argument("Renewal inversion exceeds segment optical depth");
    if (depth == 0) return u0;
    if (depth == total) return u1;
    double lo = u0, hi = u1;
    for (int iteration = 0; iteration < 100 && (hi-lo)*(end_-begin_) > tolerance; ++iteration) {
        const double middle = 0.5*lo+0.5*hi;
        if (middle == lo || middle == hi) break;
        if (integral(u0, middle) < depth) lo = middle;
        else hi = middle;
    }
    return 0.5*lo+0.5*hi;
}

RenewalRayDistribution::RenewalRayDistribution(std::vector<RenewalHazardSegment> segments, double ell)
    : segments_(std::move(segments)), ell_(ell) {
    if (segments_.empty()) throw std::invalid_argument("empty Renewal distribution");
    interval(0, 0, maximumDistance(), ell);
    prefix_.push_back(0);
    double end = 0;
    for (const auto& segment : segments_) {
        if (segment.begin() != end) throw std::invalid_argument("non-contiguous Renewal hazard segments");
        end = segment.end();
        prefix_.push_back(prefix_.back()+segment.cumulative(1));
        if (!std::isfinite(prefix_.back())) throw std::invalid_argument("Renewal cumulative optical depth overflow");
    }
}
RenewalRayDistribution RenewalRayDistribution::fromModel(const RenewalHazardModel& model,
    const RayMeanProfile& mean, const RayStartCondition& start, double ell) {
    interval(0, 0, mean.maximumX()*ell, ell);
    auto state = model.initialize(mean.segments().front(), start);
    std::vector<RenewalHazardSegment> segments;
    segments.reserve(mean.segments().size());
    for (const auto& segment : mean.segments()) {
        auto step = model.evaluate(state, segment);
        segments.emplace_back(segment.begin, segment.end, step.rates);
        state = std::move(step.nextState);
    }
    return RenewalRayDistribution(std::move(segments), ell);
}
double RenewalRayDistribution::normalized(double s) const {
    interval(s, s, maximumDistance(), ell_);
    return std::min(segments_.back().end(), s/ell_);
}
std::size_t RenewalRayDistribution::containing(double x) const {
    const auto it = std::upper_bound(segments_.begin(), segments_.end(), x,
        [](double value, const RenewalHazardSegment& segment) { return value < segment.end(); });
    return std::min(static_cast<std::size_t>(it-segments_.begin()), segments_.size()-1);
}
double RenewalRayDistribution::cumulativeHazard(double distance) const {
    const double x = normalized(distance);
    const std::size_t i = containing(x);
    return prefix_[i]+segments_[i].cumulative(local(segments_[i], x));
}
double RenewalRayDistribution::hazard(double distance) const {
    const double x = normalized(distance);
    const auto& segment = segments_[containing(x)];
    return segment.hazard(local(segment, x))/ell_;
}
double RenewalRayDistribution::transmittance(double distance) const {
    return std::exp(-cumulativeHazard(distance));
}
double RenewalRayDistribution::transmittance(double from, double to) const {
    interval(from, to, maximumDistance(), ell_);
    const double lo = normalized(from), hi = normalized(to);
    double depth = 0;
    for (std::size_t i = containing(lo); i < segments_.size() && segments_[i].begin() < hi; ++i)
        depth += segments_[i].integral(local(segments_[i], lo), local(segments_[i], hi));
    return std::exp(-depth);
}
RenewalDistanceSample RenewalRayDistribution::sample(Random& rng) const {
    return sample(rng, 0, maximumDistance());
}
RenewalDistanceSample RenewalRayDistribution::sample(Random& rng, double from, double to, double tolerance) const {
    return sampleOpticalDepth(-std::log(rng.openUniform01()), from, to, tolerance);
}
RenewalDistanceSample RenewalRayDistribution::sampleOpticalDepth(double depth, double from, double to,
                                                               double tolerance) const {
    interval(from, to, maximumDistance(), ell_);
    sampleArguments(depth, tolerance);
    const double lo = normalized(from), hi = normalized(to);
    double distance = to;
    for (std::size_t i = containing(lo); i < segments_.size() && segments_[i].begin() < hi; ++i) {
        if (consume(segments_[i], lo, hi, ell_, tolerance, depth, distance))
            return {true, std::clamp(distance, from, to), i};
    }
    return {false, to, segments_.size()};
}

RenewalDistanceSample sampleRenewalDistance(const RenewalHazardModel& model,
    const RayMeanProfile& mean, const RayStartCondition& start, double ell,
    Random& rng, double from, double to, double tolerance) {
    interval(from, to, mean.maximumX()*ell, ell);
    sampleArguments(0, tolerance);
    if (from == to) return {false, to, mean.segments().size()};
    auto state = model.initialize(mean.segments().front(), start);
    double depth = -std::log(rng.openUniform01()), distance = to;
    for (std::size_t i = 0; i < mean.segments().size(); ++i) {
        const auto& segment = mean.segments()[i];
        if (segment.begin >= to/ell) break;
        auto step = model.evaluate(state, segment);
        const RenewalHazardSegment hazard(segment.begin, segment.end, step.rates);
        if (segment.end > from/ell && consume(hazard, from/ell, to/ell, ell, tolerance, depth, distance))
            return {true, std::clamp(distance, from, to), i};
        state = std::move(step.nextState);
    }
    return {false, to, mean.segments().size()};
}
double renewalTransmittance(const RenewalHazardModel& model, const RayMeanProfile& mean,
    const RayStartCondition& start, double ell, double from, double to) {
    interval(from, to, mean.maximumX()*ell, ell);
    if (from == to) return 1;
    auto state = model.initialize(mean.segment(0), start);
    double depth = 0;
    for (std::size_t i = 0; mean.hasSegment(i); ++i) {
        const auto& segment = mean.segment(i);
        if (segment.begin >= to/ell) break;
        auto step = model.evaluate(state, segment);
        const RenewalHazardSegment hazard(segment.begin, segment.end, step.rates);
        if (segment.end > from/ell)
            depth += hazard.integral(local(hazard, from/ell), local(hazard, to/ell));
        state = std::move(step.nextState);
        if (segment.end >= to/ell) break;
    }
    return std::exp(-depth);
}

} // namespace mf

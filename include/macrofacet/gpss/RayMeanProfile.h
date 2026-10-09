#pragma once

#include "macrofacet/gpss/MeanField.h"
#include "macrofacet/mathutility/CubicPolynomial.h"
#include <array>
#include <vector>

namespace mf {

// Distances and values are normalized: x=s/ell, b=m/sigma.
struct RayMeanSegment {
    double begin = 0.0, end = 0.0;
    HermitePolynomial polynomial;
    double value(double x) const;
    double derivative(double x) const; // db/dx, one-sided inside this segment
    RayMeanSegment restricted(double lo, double hi) const;
    // [b(0), b(1), dx*b'(begin+), dx*b'(end-), log(dx)]
    std::array<double, 5> features() const;
};

class RayMeanProfile {
public:
    explicit RayMeanProfile(std::vector<RayMeanSegment> segments);
    static RayMeanProfile affine(double value, double slope, double maximumX,
                                  double maximumStep = 0.25);
    // Exact (up to roundoff) for trilinear grids and affine means. Other
    // analytic means use a piecewise cubic approximation at maximumStep.
    // Stored grids must cover the full ray. Direction must be a unit vector.
    static RayMeanProfile fromField(const MeanField& mean, const Point3& origin,
        const Vector3& direction, double sigma, double ell,
        double maximumDistance, double maximumStep = 0.25);

    // Approximate: query value/gradient at each visited segment start and use
    // its tangent within that segment. Copies have independent lazy caches.
    // A profile belongs to one ray/worker; concurrent cache mutation is unsupported.
    static RayMeanProfile pointLinear(std::shared_ptr<const MeanField> mean,
        const Point3& origin, const Vector3& direction, double sigma, double ell,
        double maximumDistance, double maximumStep = 0.25);
    bool hasSegment(std::size_t index) const;
    const RayMeanSegment& segment(std::size_t index) const;
    std::size_t constructedSegmentCount() const { return segments_.size(); }
    const std::optional<MeanJet>& initialPointJet() const { return initialPointJet_; }

    // Explicitly materializes the whole ray for callers that need a full profile.
    const std::vector<RayMeanSegment>& segments() const;
    double maximumX() const { return maximumX_; }
    // Preserves all existing cell boundaries. Reference sampling can use a
    // finer partition than the future neural model without changing b(x).
    RayMeanProfile subdivided(double maximumStep) const;
private:
    RayMeanProfile() = default;
    struct PointSource;
    std::shared_ptr<const PointSource> pointSource_;
    mutable std::vector<RayMeanSegment> segments_;
    mutable std::optional<MeanJet> initialPointJet_;
    double maximumX_ = 0;
};

} // namespace mf

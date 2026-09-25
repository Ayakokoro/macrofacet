#pragma once

#include "macrofacet/fields/ScalarField.h"
#include <array>
#include <vector>

namespace mf {

struct DensityMajorantSegment {
    double begin = 0.0;
    double end = 0.0;
    double densityMaximum = 0.0;
};

// A direction-independent macrocell bound for the actual interpolated density.
// The projected-area bound is applied when a flight direction is known.
class DensityMajorantGrid {
public:
    DensityMajorantGrid(const ScalarField& density, Bounds3 domain, int resolution = 64);
    std::vector<DensityMajorantSegment> segments(const Ray& ray,
                                                 double begin, double end) const;
    const Bounds3& domain() const { return domain_; }
private:
    std::size_t index(int i, int j, int k) const;
    Bounds3 domain_;
    int resolution_ = 0;
    Vector3 cellSize_ = Vector3::Zero();
    std::vector<double> maxima_;
};

} // namespace mf

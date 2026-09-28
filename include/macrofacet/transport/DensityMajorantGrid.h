#pragma once

#include "macrofacet/fields/ScalarField.h"
#include <array>
#include <optional>
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
    class Cursor {
    public:
        Cursor(const DensityMajorantGrid& grid, const Ray& ray, double begin, double end);
        std::optional<DensityMajorantSegment> next();
    private:
        const DensityMajorantGrid& grid_;
        Ray ray_;
        double age_ = 0.0, stop_ = 0.0;
        int cell_[3]{}, step_[3]{};
        double crossing_[3]{};
        int count_ = 0;
        double boundaryTime(int axis, int cell, int direction) const;
    };

    DensityMajorantGrid(const ScalarField& density, Bounds3 domain, int resolution = 64);
    Cursor sampleRay(const Ray& ray, double begin, double end) const {
        return Cursor(*this, ray, begin, end);
    }
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

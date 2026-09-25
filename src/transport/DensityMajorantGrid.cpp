#include "macrofacet/transport/DensityMajorantGrid.h"
#include "macrofacet/mathutility/NumericPolicy.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {

std::size_t DensityMajorantGrid::index(int i, int j, int k) const {
    return (static_cast<std::size_t>(k) * resolution_ + j) * resolution_ + i;
}

DensityMajorantGrid::DensityMajorantGrid(const ScalarField& density, Bounds3 domain,
                                         int resolution)
    : domain_(std::move(domain)), resolution_(resolution) {
    if (!domain_.valid() || resolution_ < 1 || resolution_ > 256)
        throw std::invalid_argument("invalid density majorant grid");
    cellSize_ = (domain_.maximum - domain_.minimum) / resolution_;
    maxima_.resize(static_cast<std::size_t>(resolution_) * resolution_ * resolution_);
    for (int k = 0; k < resolution_; ++k)
        for (int j = 0; j < resolution_; ++j)
            for (int i = 0; i < resolution_; ++i) {
                const Point3 lo = domain_.minimum +
                    cellSize_.cwiseProduct(Point3(i, j, k));
                const Point3 hi = domain_.minimum +
                    cellSize_.cwiseProduct(Point3(i + 1, j + 1, k + 1));
                const ScalarBounds bounds = density.bounds({lo, hi});
                if (!bounds.certified || !(bounds.maximumValue >= 0.0) ||
                    !std::isfinite(bounds.maximumValue))
                    throw NumericError(NumericStatus::InvalidMajorant,
                                       "density macrocell lacks a finite certified bound");
                maxima_[index(i, j, k)] = bounds.maximumValue == 0.0 ? 0.0 :
                    std::nextafter(bounds.maximumValue,
                                   std::numeric_limits<double>::infinity());
            }
}

std::vector<DensityMajorantSegment> DensityMajorantGrid::segments(
    const Ray& ray, double begin, double end) const {
    std::vector<DensityMajorantSegment> result;
    const DomainInterval interval = domain_.intersect(ray);
    if (!interval.hit) return result;
    double age = std::max(begin, interval.entry);
    const double stop = std::min(end, interval.exit);
    if (!(age < stop)) return result;
    const Point3 p = ray.origin + age * ray.direction;
    int cell[3], step[3];
    double next[3];
    const auto boundaryTime = [&](int axis, int cellIndex, int direction) {
        const int boundary = cellIndex + (direction > 0 ? 1 : 0);
        // Use the exact domain face at the outermost boundary. Recomputing
        // times from planes also avoids accumulated error from repeated DDA
        // increments, which can otherwise put the last boundary before exit.
        const double plane = boundary == 0 ? domain_.minimum[axis] :
            boundary == resolution_ ? domain_.maximum[axis] :
            domain_.minimum[axis] + boundary * cellSize_[axis];
        return (plane - ray.origin[axis]) / ray.direction[axis];
    };
    for (int axis = 0; axis < 3; ++axis) {
        const double u = (p[axis] - domain_.minimum[axis]) / cellSize_[axis];
        int idx = static_cast<int>(std::floor(u));
        const double rounded = std::round(u);
        if (ray.direction[axis] < 0.0 &&
            std::abs(u - rounded) <= 16.0 * std::numeric_limits<double>::epsilon() *
                                      std::max(1.0, std::abs(u)))
            --idx;
        cell[axis] = std::clamp(idx, 0, resolution_ - 1);
        step[axis] = ray.direction[axis] > 0.0 ? 1 : ray.direction[axis] < 0.0 ? -1 : 0;
        if (step[axis] == 0) {
            next[axis] = std::numeric_limits<double>::infinity();
        } else {
            next[axis] = boundaryTime(axis, cell[axis], step[axis]);
            if (next[axis] <= age) next[axis] = std::nextafter(age,
                                                std::numeric_limits<double>::infinity());
        }
    }
    for (int count = 0; age < stop; ++count) {
        if (count > 3 * resolution_ + 8)
            throw NumericError(NumericStatus::InvalidMajorant,
                               "density majorant DDA exceeded its grid traversal budget");
        const double boundary = std::min({next[0], next[1], next[2]});
        const double segmentEnd = std::min(stop, boundary);
        if (!(segmentEnd > age))
            throw NumericError(NumericStatus::InvalidMajorant,
                               "density majorant DDA failed to advance");
        result.push_back({age, segmentEnd, maxima_[index(cell[0], cell[1], cell[2])]});
        age = segmentEnd;
        if (!(age < stop)) break;
        for (int axis = 0; axis < 3; ++axis) {
            if (next[axis] <= boundary) {
                cell[axis] += step[axis];
                if (cell[axis] < 0 || cell[axis] >= resolution_)
                    throw NumericError(NumericStatus::InvalidMajorant,
                                       "density majorant DDA left its domain early");
                next[axis] = boundaryTime(axis, cell[axis], step[axis]);
                if (next[axis] <= age) next[axis] = std::nextafter(age,
                                                    std::numeric_limits<double>::infinity());
            }
        }
    }
    return result;
}

} // namespace mf

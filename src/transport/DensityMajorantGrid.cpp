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

double DensityMajorantGrid::Cursor::boundaryTime(int axis, int cellIndex,
                                                 int direction) const {
        const int boundary = cellIndex + (direction > 0 ? 1 : 0);
        // Use the exact domain face at the outermost boundary. Recomputing
        // times from planes also avoids accumulated error from repeated DDA
        // increments, which can otherwise put the last boundary before exit.
        const double plane = boundary == 0 ? grid_.domain_.minimum[axis] :
            boundary == grid_.resolution_ ? grid_.domain_.maximum[axis] :
            grid_.domain_.minimum[axis] + boundary * grid_.cellSize_[axis];
        return (plane - ray_.origin[axis]) / ray_.direction[axis];
}

DensityMajorantGrid::Cursor::Cursor(const DensityMajorantGrid& grid, const Ray& ray,
                                    double begin, double end)
    : grid_(grid), ray_(ray) {
    const DomainInterval interval = grid_.domain_.intersect(ray_);
    if (!interval.hit) return;
    age_ = std::max(begin, interval.entry);
    stop_ = std::min(end, interval.exit);
    if (!(age_ < stop_)) return;
    const Point3 p = ray_.origin + age_ * ray_.direction;
    for (int axis = 0; axis < 3; ++axis) {
        const double u = (p[axis] - grid_.domain_.minimum[axis]) / grid_.cellSize_[axis];
        int idx = static_cast<int>(std::floor(u));
        const double rounded = std::round(u);
        if (ray_.direction[axis] < 0.0 &&
            std::abs(u - rounded) <= 16.0 * std::numeric_limits<double>::epsilon() *
                                      std::max(1.0, std::abs(u)))
            --idx;
        cell_[axis] = std::clamp(idx, 0, grid_.resolution_ - 1);
        step_[axis] = ray_.direction[axis] > 0.0 ? 1 : ray_.direction[axis] < 0.0 ? -1 : 0;
        if (step_[axis] == 0) {
            crossing_[axis] = std::numeric_limits<double>::infinity();
        } else {
            crossing_[axis] = boundaryTime(axis, cell_[axis], step_[axis]);
            if (crossing_[axis] <= age_) crossing_[axis] = std::nextafter(age_,
                                                std::numeric_limits<double>::infinity());
        }
    }
}

std::optional<DensityMajorantSegment> DensityMajorantGrid::Cursor::next() {
    if (!(age_ < stop_)) return std::nullopt;
        if (count_++ > 3 * grid_.resolution_ + 8)
            throw NumericError(NumericStatus::InvalidMajorant,
                               "density majorant DDA exceeded its grid traversal budget");
        const double boundary = std::min({crossing_[0], crossing_[1], crossing_[2]});
        const double segmentEnd = std::min(stop_, boundary);
        if (!(segmentEnd > age_))
            throw NumericError(NumericStatus::InvalidMajorant,
                               "density majorant DDA failed to advance");
        const DensityMajorantSegment segment{age_, segmentEnd,
            grid_.maxima_[grid_.index(cell_[0], cell_[1], cell_[2])]};
        age_ = segmentEnd;
        if (!(age_ < stop_)) return segment;
        for (int axis = 0; axis < 3; ++axis) {
            if (crossing_[axis] <= boundary) {
                cell_[axis] += step_[axis];
                if (cell_[axis] < 0 || cell_[axis] >= grid_.resolution_)
                    throw NumericError(NumericStatus::InvalidMajorant,
                                       "density majorant DDA left its domain early");
                crossing_[axis] = boundaryTime(axis, cell_[axis], step_[axis]);
                if (crossing_[axis] <= age_) crossing_[axis] = std::nextafter(age_,
                                                    std::numeric_limits<double>::infinity());
            }
        }
        return segment;
}

std::vector<DensityMajorantSegment> DensityMajorantGrid::segments(
    const Ray& ray, double begin, double end) const {
    std::vector<DensityMajorantSegment> result;
    Cursor cursor(*this, ray, begin, end);
    while (const auto segment = cursor.next()) result.push_back(*segment);
    return result;
}

} // namespace mf

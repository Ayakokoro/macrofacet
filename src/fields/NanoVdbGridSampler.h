#pragma once

// Internal helper shared by NanoVdbMean and NanoVdbSampledField. It is the only
// place that talks to the nanovdb API on the render side. Both bakers now store
// each sample at NanoVDB's indexToWorld(ijk) = origin + dx * ijk, matching
// primitive_macrofacet. The continuous index is (P - origin) / dx.

#include "macrofacet/core/Types.h"
#include "macrofacet/gpss/MeanField.h"
#include "fields/TrilinearBounds.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace mf::detail {

class GridView {
public:
    // Reads `gridName` out of a .nvdb and caches everything the sampler needs.
    // Throws when the file or the named grid is missing, or when it is not a
    // float grid.
    static GridView open(const std::filesystem::path& path, const std::string& gridName) {
        GridView view;
        try {
            view.handle_ = nanovdb::io::readGrid<nanovdb::HostBuffer>(path.string(), gridName);
        } catch (const std::exception& error) {
            throw std::runtime_error("cannot read grid \"" + gridName + "\" from " + path.string() +
                                     ": " + error.what());
        }
        view.grid_ = view.handle_.grid<float>();
        if (!view.grid_) {
            throw std::runtime_error("grid \"" + gridName + "\" in " + path.string() +
                                     " is not a float grid");
        }
        view.dx_ = view.grid_->voxelSize()[0];
        if (!(view.dx_ > 0.0) || !std::isfinite(view.dx_)) {
            throw std::runtime_error("grid \"" + gridName + "\" has a degenerate voxel size");
        }
        const nanovdb::Vec3d origin = view.grid_->map().applyMap(nanovdb::Vec3d(0.0));
        view.origin_ = Point3(origin[0], origin[1], origin[2]);
        view.background_ = view.grid_->tree().root().background();

        // Active-value extremes. updateGridStats walks the tree once; the root
        // then carries the min/max of the active values.
        nanovdb::tools::updateGridStats(view.handle_.grid<float>());
        view.minimum_ = static_cast<double>(view.grid_->tree().root().minimum());
        view.maximum_ = static_cast<double>(view.grid_->tree().root().maximum());
        if (!(view.maximum_ >= view.minimum_)) {
            // An empty tree (the constant-grid case) reports both as zero.
            view.minimum_ = view.maximum_ = static_cast<double>(view.background_);
        }
        // A sub-domain's extremes are a subset of the grid's, so folding the
        // background in keeps bounds() conservative for both narrow-band and
        // full-domain fields.
        view.maximum_ = std::max(view.maximum_, static_cast<double>(view.background_));
        view.minimum_ = std::min(view.minimum_, static_cast<double>(view.background_));

        const nanovdb::CoordBBox bbox = view.grid_->indexBBox();
        view.worldBounds_.minimum = view.origin_ + view.dx_ * Point3(bbox.min()[0], bbox.min()[1],
                                                                     bbox.min()[2]);
        view.worldBounds_.maximum = view.origin_ + view.dx_ * Point3(bbox.max()[0] + 1,
                                                                     bbox.max()[1] + 1,
                                                                     bbox.max()[2] + 1);
        view.activeNodeBounds_.minimum = view.worldBounds_.minimum;
        view.activeNodeBounds_.maximum = view.origin_ + view.dx_ * Point3(bbox.max()[0],
                                                                          bbox.max()[1],
                                                                          bbox.max()[2]);
        return view;
    }

    double voxelSize() const { return dx_; }
    const Point3& origin() const { return origin_; }
    double background() const { return static_cast<double>(background_); }
    double minimumValue() const { return minimum_; }
    double maximumValue() const { return maximum_; }
    const Bounds3& worldBounds() const { return worldBounds_; }
    const Bounds3& activeNodeBounds() const { return activeNodeBounds_; }

    bool hasCompleteInterpolationCell(const Point3& x) const {
        const auto u = continuousIndex(x);
        const auto bbox = grid_->indexBBox();
        int base[3];
        for (int axis = 0; axis < 3; ++axis) {
            // Check bounds before conversion to avoid overflow for far queries.
            if (!std::isfinite(u[axis]) || u[axis] < bbox.min()[axis] ||
                u[axis] >= bbox.max()[axis]) return false;
            base[axis] = static_cast<int>(std::floor(u[axis]));
        }
        auto accessor = grid_->getAccessor();
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    const nanovdb::Coord node(base[0] + i, base[1] + j, base[2] + k);
                    if (!accessor.isActive(node) || !std::isfinite(accessor.getValue(node)))
                        return false;
                }
        return true;
    }

    // With shared trilinear weights, positive density can only occur where
    // alpha is positive if that implication holds at every grid node.
    bool positiveAtEveryPositiveNodeOf(const GridView& density) const {
        if (dx_ != density.dx_ || (origin_ - density.origin_).squaredNorm() != 0.0 ||
            density.background_ != 0.0f || background_ != 0.0f)
            return false;
        const nanovdb::CoordBBox active = density.grid_->indexBBox();
        auto densityAccessor = density.grid_->getAccessor();
        auto alphaAccessor = grid_->getAccessor();
        for (int k = active.min()[2]; k <= active.max()[2]; ++k)
            for (int j = active.min()[1]; j <= active.max()[1]; ++j)
                for (int i = active.min()[0]; i <= active.max()[0]; ++i) {
                    const nanovdb::Coord node(i, j, k);
                    const float d = densityAccessor.getValue(node);
                    const float a = alphaAccessor.getValue(node);
                    if (!std::isfinite(d) || !std::isfinite(a) || d < 0.0f || a < 0.0f ||
                        (d > 0.0f && !(a > 0.0f))) return false;
                }
        return true;
    }

    // Scan each interpolation cell touching an active voxel once. A cell
    // outside this range has eight background corners and zero gradient.
    // This is a single global bound, computed when NanoVdbMean is opened.
    double maximumTrilinearGradientNorm() const {
        const nanovdb::CoordBBox active = grid_->indexBBox();
        auto accessor = grid_->getAccessor();
        double maximum = 0.0;
        for (int k = active.min()[2] - 1; k <= active.max()[2]; ++k) {
            for (int j = active.min()[1] - 1; j <= active.max()[1]; ++j) {
                for (int i = active.min()[0] - 1; i <= active.max()[0]; ++i) {
                    double corner[2][2][2];
                    for (int dk = 0; dk < 2; ++dk)
                        for (int dj = 0; dj < 2; ++dj)
                            for (int di = 0; di < 2; ++di)
                                corner[di][dj][dk] = static_cast<double>(accessor.getValue(
                                    nanovdb::Coord(i + di, j + dj, k + dk)));
                    maximum = std::max(maximum,
                        trilinearCellGradientNormBound(corner, dx_));
                }
            }
        }
        return maximum;
    }

    // Trilinear interpolation is a convex combination of its eight corner
    // samples. Every corner touched by a world-space box is scanned, including
    // inactive/background values. This certifies local scalar majorants.
    std::pair<double, double> bounds(const Bounds3& box) const {
        if (!box.valid()) throw std::invalid_argument("invalid grid bounds query");
        const nanovdb::Vec3d low = continuousIndex(box.minimum);
        const nanovdb::Vec3d high = continuousIndex(box.maximum);
        const nanovdb::CoordBBox active = grid_->indexBBox();
        int lo[3], hi[3];
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::max(static_cast<int>(std::floor(low[axis])), active.min()[axis]);
            hi[axis] = std::min(static_cast<int>(std::floor(high[axis])) + 1,
                                active.max()[axis]);
        }
        double minimum = background_, maximum = background_;
        if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2])
            return {minimum, maximum};
        auto accessor = grid_->getAccessor();
        for (int k = lo[2]; k <= hi[2]; ++k)
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    const double value = accessor.getValue(nanovdb::Coord(i, j, k));
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
        return {minimum, maximum};
    }

    // Trilinear value at a world point; the background outside the grid.
    MeanRayBounds rayCellBounds(const Point3& origin, const Vector3& direction,
                                 double begin, double end) const {
        const auto middle = continuousIndex(origin + (0.5 * begin + 0.5 * end) * direction);
        int base[3];
        double low[3], high[3], first[3];
        const auto start = continuousIndex(origin + begin * direction);
        const auto stop = continuousIndex(origin + end * direction);
        double coordinateScale = 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            base[axis] = static_cast<int>(std::floor(middle[axis]));
            first[axis] = start[axis] - base[axis];
            low[axis] = std::clamp(std::min(start[axis], stop[axis]) - base[axis], 0.0, 1.0);
            high[axis] = std::clamp(std::max(start[axis], stop[axis]) - base[axis], 0.0, 1.0);
            coordinateScale = std::max(coordinateScale,
                (std::abs(origin[axis]) + std::abs(origin_[axis]) +
                 std::abs(end * direction[axis])) / dx_);
        }
        auto accessor = grid_->getAccessor();
        double c[2][2][2], magnitude = 0.0;
        for (int k = 0; k < 2; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
            c[i][j][k] = accessor.getValue(nanovdb::Coord(base[0]+i, base[1]+j, base[2]+k));
            magnitude = std::max(magnitude, std::abs(c[i][j][k]));
        }
        const auto jet = [&](const double* f) {
            MeanJet out;
            for (int k = 0; k < 2; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
                const double x = i ? f[0] : 1-f[0], y = j ? f[1] : 1-f[1], z = k ? f[2] : 1-f[2];
                const double value = c[i][j][k];
                out.value += x*y*z*value;
                out.gradient += value / dx_ * Vector3((i ? 1.0 : -1.0)*y*z,
                    x*(j ? 1.0 : -1.0)*z, x*y*(k ? 1.0 : -1.0));
            }
            return out;
        };
        MeanRayBounds result;
        result.minimumValue = result.minimumDerivative = std::numeric_limits<double>::infinity();
        result.maximumValue = result.maximumDerivative = -std::numeric_limits<double>::infinity();
        // Value and directional derivative are multi-affine on this box.
        for (int k = 0; k < 2; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
            const double f[3]{i ? high[0] : low[0], j ? high[1] : low[1], k ? high[2] : low[2]};
            const MeanJet value = jet(f);
            result.minimumValue = std::min(result.minimumValue, value.value);
            result.maximumValue = std::max(result.maximumValue, value.value);
            result.minimumDerivative = std::min(result.minimumDerivative, value.gradient.dot(direction));
            result.maximumDerivative = std::max(result.maximumDerivative, value.gradient.dot(direction));
        }
        double xy=0.0, xz=0.0, yz=0.0;
        for (int i=0; i<2; ++i) {
            xy = std::max(xy, std::abs(c[1][1][i]-c[1][0][i]-c[0][1][i]+c[0][0][i]));
            xz = std::max(xz, std::abs(c[1][i][1]-c[1][i][0]-c[0][i][1]+c[0][i][0]));
            yz = std::max(yz, std::abs(c[i][1][1]-c[i][1][0]-c[i][0][1]+c[i][0][0]));
        }
        result.maximumSecondDerivative = 2.0 / (dx_*dx_) *
            (std::abs(direction.x()*direction.y())*xy +
             std::abs(direction.x()*direction.z())*xz + std::abs(direction.y()*direction.z())*yz);
        result.beginDerivative = jet(first).gradient.dot(direction);
        const double error = 256.0 * std::numeric_limits<double>::epsilon() *
            (1.0 + coordinateScale) * std::max(magnitude, 1e-30);
        result.minimumValue -= error;
        result.maximumValue += error;
        result.minimumDerivative -= error * direction.lpNorm<1>() / dx_;
        result.maximumDerivative += error * direction.lpNorm<1>() / dx_;
        result.maximumSecondDerivative += error * direction.squaredNorm() / (dx_*dx_);
        result.certified = true;
        return result;
    }

    double valueDifference(const Point3& x, const Vector3& displacement) const {
        const auto u = continuousIndex(x);
        int base[3];
        double f[3], df[3];
        for (int a=0; a<3; ++a) {
            base[a] = static_cast<int>(std::floor(u[a]));
            f[a] = u[a]-base[a];
            df[a] = displacement[a]/dx_;
            if (f[a]+df[a] < 0.0 || f[a]+df[a] > 1.0)
                return sample(x+displacement)-sample(x);
        }
        auto accessor = grid_->getAccessor();
        double difference=0.0;
        for (int k=0; k<2; ++k) for (int j=0; j<2; ++j) for (int i=0; i<2; ++i) {
            const double x0=i?f[0]:1-f[0], y0=j?f[1]:1-f[1], z0=k?f[2]:1-f[2];
            const double xd=(i?1:-1)*df[0], yd=(j?1:-1)*df[1], zd=(k?1:-1)*df[2];
            difference += (xd*y0*z0+(x0+xd)*yd*z0+(x0+xd)*(y0+yd)*zd) *
                accessor.getValue(nanovdb::Coord(base[0]+i,base[1]+j,base[2]+k));
        }
        return difference;
    }

    double sample(const Point3& x) const {
        const nanovdb::Vec3d u = continuousIndex(x);
        const nanovdb::Coord base(static_cast<int>(std::floor(u[0])),
                                  static_cast<int>(std::floor(u[1])),
                                  static_cast<int>(std::floor(u[2])));
        const nanovdb::Vec3d f(u[0] - base[0], u[1] - base[1], u[2] - base[2]);
        auto accessor = grid_->getAccessor();
        double value = 0.0;
        for (int dk = 0; dk < 2; ++dk) {
            const double wz = dk ? f[2] : 1.0 - f[2];
            for (int dj = 0; dj < 2; ++dj) {
                const double wy = dj ? f[1] : 1.0 - f[1];
                for (int di = 0; di < 2; ++di) {
                    const double wx = di ? f[0] : 1.0 - f[0];
                    value += wx * wy * wz * static_cast<double>(accessor.getValue(
                        nanovdb::Coord(base[0] + di, base[1] + dj, base[2] + dk)));
                }
            }
        }
        return value;
    }

    // Trilinear value plus the gradient of that same interpolant, in world
    // units. Returning the derivative of the function we actually evaluate
    // keeps evaluate() and any caller that differentiates it consistent, and
    // costs no extra grid lookups.
    void sampleWithGradient(const Point3& x, double& value, Vector3& gradient) const {
        sampleWithGradientInCell(x, x, value, gradient);
    }

    void sampleWithGradientInCell(const Point3& x, const Point3& interior,
                                  double& value, Vector3& gradient, bool requireComplete = false) const {
        const nanovdb::Vec3d u = continuousIndex(x);
        const nanovdb::Vec3d cell = continuousIndex(interior);
        if (requireComplete) {
            const auto bbox = grid_->indexBBox();
            for (int a = 0; a < 3; ++a)
                if (!std::isfinite(cell[a]) || cell[a] < bbox.min()[a] || cell[a] >= bbox.max()[a])
                    throw std::invalid_argument("point query reaches outside stored SDF interpolation cells");
        }
        const nanovdb::Coord base(static_cast<int>(std::floor(cell[0])),
                                  static_cast<int>(std::floor(cell[1])),
                                  static_cast<int>(std::floor(cell[2])));
        const nanovdb::Vec3d f(u[0] - base[0], u[1] - base[1], u[2] - base[2]);

        auto accessor = grid_->getAccessor();
        double corner[2][2][2];
        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    const nanovdb::Coord node(base[0] + di, base[1] + dj, base[2] + dk);
                    const double sample = static_cast<double>(accessor.getValue(node));
                    if (requireComplete && (!accessor.isActive(node) || !std::isfinite(sample)))
                        throw std::invalid_argument("point query reaches missing/non-finite SDF interpolation nodes");
                    corner[di][dj][dk] = sample;
                }
            }
        }

        const double wx[2] = {1.0 - f[0], f[0]};
        const double wy[2] = {1.0 - f[1], f[1]};
        const double wz[2] = {1.0 - f[2], f[2]};
        const double dwx[2] = {-1.0, 1.0};
        const double dwy[2] = {-1.0, 1.0};
        const double dwz[2] = {-1.0, 1.0};

        value = 0.0;
        double du[3] = {0.0, 0.0, 0.0};
        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    const double v = corner[di][dj][dk];
                    value += wx[di] * wy[dj] * wz[dk] * v;
                    du[0] += dwx[di] * wy[dj] * wz[dk] * v;
                    du[1] += wx[di] * dwy[dj] * wz[dk] * v;
                    du[2] += wx[di] * wy[dj] * dwz[dk] * v;
                }
            }
        }
        const double inverseDx = 1.0 / dx_;
        gradient = Vector3(du[0] * inverseDx, du[1] * inverseDx, du[2] * inverseDx);
    }

private:
    // World point -> continuous index of the stored sample nodes.
    nanovdb::Vec3d continuousIndex(const Point3& x) const {
        return nanovdb::Vec3d((x.x() - origin_.x()) / dx_,
                              (x.y() - origin_.y()) / dx_,
                              (x.z() - origin_.z()) / dx_);
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> handle_;
    const nanovdb::NanoGrid<float>* grid_ = nullptr;
    double dx_ = 1.0;
    Point3 origin_ = Point3::Zero();
    float background_ = 0.0f;
    double minimum_ = 0.0;
    double maximum_ = 0.0;
    Bounds3 worldBounds_;
    Bounds3 activeNodeBounds_;
};

} // namespace mf::detail

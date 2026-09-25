#pragma once

// Internal helper shared by NanoVdbMean and NanoVdbSampledField. It is the only
// place that talks to the nanovdb API on the render side, so the half-voxel
// convention (see docs/archive/PLAN_NANOVDB_FIELD.md 2.2) is written down exactly once.
//
// Two conventions meet here and they are NOT the same:
//
//   NanoVDB's Map is corner-based -- indexToWorld(i) == applyMap(i) == origin +
//   dx * i, with no half-voxel offset.
//
//   The value stored in voxel ijk is the field sampled at that voxel's *centre*,
//   origin + dx * (ijk + 0.5), which is OpenVDB's convention and what VdbBaker
//   writes.
//
// So a world point maps to the continuous index in the *centre* frame:
//
//   continuous index  = (P - origin) / dx - 0.5
//   base voxel        = floor(continuous index)
//   fraction          = continuous index - base
//
// Subtracting the half voxel is what makes the interpolant reproduce the stored
// value exactly at a voxel centre (u == i, fraction 0). Omitting it reads the
// field half a voxel off, which is invisible on a flat region and a constant
// bias of dx/2 * |grad| everywhere else.

#include "macrofacet/core/Types.h"

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
        // background in keeps bounds() conservative for the sdf grid, whose
        // background (+6 sigma) sits above every active value.
        view.maximum_ = std::max(view.maximum_, static_cast<double>(view.background_));
        view.minimum_ = std::min(view.minimum_, static_cast<double>(view.background_));

        const nanovdb::CoordBBox bbox = view.grid_->indexBBox();
        view.worldBounds_.minimum = view.origin_ + view.dx_ * Point3(bbox.min()[0], bbox.min()[1],
                                                                     bbox.min()[2]);
        view.worldBounds_.maximum = view.origin_ + view.dx_ * Point3(bbox.max()[0] + 1,
                                                                     bbox.max()[1] + 1,
                                                                     bbox.max()[2] + 1);
        return view;
    }

    double voxelSize() const { return dx_; }
    const Point3& origin() const { return origin_; }
    double background() const { return static_cast<double>(background_); }
    double minimumValue() const { return minimum_; }
    double maximumValue() const { return maximum_; }
    const Bounds3& worldBounds() const { return worldBounds_; }

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
        const nanovdb::Vec3d u = continuousIndex(x);
        const nanovdb::Coord base(static_cast<int>(std::floor(u[0])),
                                  static_cast<int>(std::floor(u[1])),
                                  static_cast<int>(std::floor(u[2])));
        const nanovdb::Vec3d f(u[0] - base[0], u[1] - base[1], u[2] - base[2]);

        auto accessor = grid_->getAccessor();
        double corner[2][2][2];
        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    corner[di][dj][dk] = static_cast<double>(accessor.getValue(
                        nanovdb::Coord(base[0] + di, base[1] + dj, base[2] + dk)));
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
    // World point -> continuous index in the centre frame. See the note at the
    // top of this file for why the half voxel is subtracted.
    nanovdb::Vec3d continuousIndex(const Point3& x) const {
        return nanovdb::Vec3d((x.x() - origin_.x()) / dx_ - 0.5,
                              (x.y() - origin_.y()) / dx_ - 0.5,
                              (x.z() - origin_.z()) / dx_ - 0.5);
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> handle_;
    const nanovdb::NanoGrid<float>* grid_ = nullptr;
    double dx_ = 1.0;
    Point3 origin_ = Point3::Zero();
    float background_ = 0.0f;
    double minimum_ = 0.0;
    double maximum_ = 0.0;
    Bounds3 worldBounds_;
};

} // namespace mf::detail

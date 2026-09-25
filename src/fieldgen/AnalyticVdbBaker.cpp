#include "fieldgen/VdbBaker.h"
#include "fields/TrilinearBounds.h"
#include "macrofacet/mathutility/Gaussian1D.h"

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {

BakedGrids bakeAnalyticMean(const MeanField& mean, const Bounds3& domain,
                           double dx, double sigma, double alpha) {
    if (!domain.valid() || !(dx > 0.0) || !(sigma > 0.0) || !(alpha > 0.0) ||
        !std::isfinite(dx) || !std::isfinite(sigma) || !std::isfinite(alpha)) {
        throw std::invalid_argument("invalid analytic field bake parameters");
    }
    const Vector3 extent = domain.maximum - domain.minimum;
    std::array<int, 3> dimensions{};
    long long voxelCount = 1;
    for (int axis = 0; axis < 3; ++axis) {
        const double count = std::ceil(extent[axis] / dx) + 3.0;
        if (!(count >= 3.0) || count > 2048.0 ||
            voxelCount > 8000000LL / static_cast<long long>(count)) {
            throw std::invalid_argument(
                "analytic NVDB bake exceeds 8 million voxels; increase field.bake_voxel_size");
        }
        dimensions[axis] = static_cast<int>(count);
        voxelCount *= dimensions[axis];
    }

    // The first and last sample centres lie outside the active domain. Every
    // trilinear cell touched by tracing is therefore backed by real samples.
    const Point3 origin = domain.minimum - Vector3::Constant(1.5 * dx);
    const float background = static_cast<float>(6.0 * sigma);
    nanovdb::tools::build::Grid<float> sdfGrid(background, "sdf",
                                               nanovdb::GridClass::LevelSet);
    nanovdb::tools::build::Grid<float> densityGrid(0.0f, "density",
                                                   nanovdb::GridClass::Unknown);
    nanovdb::tools::build::Grid<float> alphaGrid(static_cast<float>(alpha), "alpha",
                                                 nanovdb::GridClass::Unknown);
    const nanovdb::Vec3d mapOrigin(origin.x(), origin.y(), origin.z());
    sdfGrid.setTransform(dx, mapOrigin);
    densityGrid.setTransform(dx, mapOrigin);
    alphaGrid.setTransform(dx, mapOrigin);

    BakeReport report;
    report.dx = dx;
    report.origin = origin;
    report.domainMinimum = domain.minimum;
    report.domainMaximum = domain.maximum;
    report.voxels = {dimensions[0], dimensions[1], dimensions[2]};
    report.background = background;
    report.threads = 1;
    report.sdfMinimum = std::numeric_limits<double>::infinity();
    report.sdfMaximum = -std::numeric_limits<double>::infinity();

    for (int k = 0; k < dimensions[2]; ++k) {
        for (int j = 0; j < dimensions[1]; ++j) {
            for (int i = 0; i < dimensions[0]; ++i) {
                const Point3 x = origin + dx * Point3(i + 0.5, j + 0.5, k + 0.5);
                double value;
                try {
                    value = mean.evaluate(x).value;
                } catch (const std::domain_error&) {
                    // Some exact SDFs have an undefined derivative at isolated
                    // points (for example a sphere centre). Sample a tiny
                    // symmetric offset to obtain the continuous value there.
                    const Vector3 offset = Vector3::UnitX() * (dx * 1e-5);
                    value = 0.5 * (mean.evaluate(x - offset).value +
                                   mean.evaluate(x + offset).value);
                }
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("analytic mean returned a non-finite value during bake");
                }
                const float stored = static_cast<float>(value);
                if (!std::isfinite(stored)) {
                    throw std::invalid_argument("analytic mean exceeds float32 range during bake");
                }
                const float density = static_cast<float>(normalPdfOverCdf(value / sigma) / sigma);
                if (!std::isfinite(density)) {
                    throw std::invalid_argument("analytic density exceeds float32 range during bake");
                }
                const nanovdb::Coord ijk(i, j, k);
                sdfGrid.setValue(ijk, stored);
                densityGrid.setValue(ijk, density);
                alphaGrid.setValue(ijk, static_cast<float>(alpha));
                report.sdfMinimum = std::min(report.sdfMinimum, static_cast<double>(stored));
                report.sdfMaximum = std::max(report.sdfMaximum, static_cast<double>(stored));
                report.densityMaximum = std::max(report.densityMaximum, static_cast<double>(density));
            }
        }
    }
    report.bandVoxels = voxelCount;
    report.maximumGradientNorm = detail::trilinearGradientNormBound(
        std::min(report.sdfMinimum, static_cast<double>(background)),
        std::max(report.sdfMaximum, static_cast<double>(background)), dx);

    nanovdb::tools::build::Grid<float> sigmaGrid(static_cast<float>(sigma), "sigma",
                                                 nanovdb::GridClass::Unknown);
    sigmaGrid.setTransform(dx, mapOrigin);
    BakedGrids result;
    result.grids.resize(kGridCount);
    result.grids[kDensityGrid] = nanovdb::tools::createNanoGrid(densityGrid);
    result.grids[kAlphaGrid] = nanovdb::tools::createNanoGrid(alphaGrid);
    result.grids[kSdfGrid] = nanovdb::tools::createNanoGrid(sdfGrid);
    result.grids[kSigmaGrid] = nanovdb::tools::createNanoGrid(sigmaGrid);
    result.report = report;
    return result;
}

} // namespace mf

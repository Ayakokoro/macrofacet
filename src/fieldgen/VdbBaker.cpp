#include "fieldgen/VdbBaker.h"
#include "fields/TrilinearBounds.h"

#include "macrofacet/mathutility/Gaussian1D.h"

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mf {
namespace {

int resolveWorkerCount(int requested, int slabs) {
    if (slabs < 1) return 1;
    if (requested > 0) return std::max(1, std::min(requested, slabs));
    const unsigned hardware = std::thread::hardware_concurrency();
    const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
    return std::max(1, std::min(available, slabs));
}

// One voxel that landed inside the narrow band. Collected per worker so the
// parallel phase touches no shared state; the grids are filled single-threaded
// afterwards, because nanovdb's build::Grid is not thread-safe for writes.
struct BandVoxel {
    int i = 0, j = 0, k = 0;
    float sdf = 0.0f;
    float density = 0.0f;
};

void validateSettings(const BakeSettings& settings) {
    if (!(settings.sigma > 0.0) || !std::isfinite(settings.sigma)) {
        throw std::invalid_argument("sigma must be a finite positive number");
    }
    if (!(settings.alpha > 0.0) || !std::isfinite(settings.alpha)) {
        throw std::invalid_argument("alpha must be a finite positive number");
    }
    if (!(settings.bandSigmas > 0.0) || !std::isfinite(settings.bandSigmas)) {
        throw std::invalid_argument("band must be a finite positive number of sigmas");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (settings.resolution[axis] < 1) {
            throw std::invalid_argument("resolution must be at least one voxel per axis");
        }
    }
}

struct Frame {
    double dx = 0.0;
    Point3 origin = Point3::Zero();
    Point3 domainMinimum = Point3::Zero();
    Point3 domainMaximum = Point3::Zero();
    nanovdb::Coord lo{0, 0, 0};
    nanovdb::Coord hi{0, 0, 0};
};

Frame makeFrame(const TriangleMesh& mesh, const BakeSettings& settings) {
    const Bounds3 meshBounds = mesh.bounds();
    Frame frame;
    const double margin = settings.bandSigmas * settings.sigma;
    frame.domainMinimum = meshBounds.minimum.array() - margin;
    frame.domainMaximum = meshBounds.maximum.array() + margin;
    // Match macrofcaet_vdb_generator: resolution divides the expanded box,
    // while the NanoVDB map stays anchored at the world origin.
    const Vector3 gap = frame.domainMaximum - frame.domainMinimum;
    frame.dx = std::min({gap.x() / settings.resolution[0],
                         gap.y() / settings.resolution[1],
                         gap.z() / settings.resolution[2]});
    if (!(frame.dx > 0.0) || !std::isfinite(frame.dx)) {
        throw std::invalid_argument("mesh bounding box is degenerate");
    }
    frame.origin = Point3::Zero();
    frame.lo = nanovdb::Coord(static_cast<int>(std::floor(frame.domainMinimum.x() / frame.dx)),
                              static_cast<int>(std::floor(frame.domainMinimum.y() / frame.dx)),
                              static_cast<int>(std::floor(frame.domainMinimum.z() / frame.dx)));
    frame.hi = nanovdb::Coord(static_cast<int>(std::floor(frame.domainMaximum.x() / frame.dx)),
                              static_cast<int>(std::floor(frame.domainMaximum.y() / frame.dx)),
                              static_cast<int>(std::floor(frame.domainMaximum.z() / frame.dx)));
    return frame;
}

} // namespace

double macrofacetDensity(double distance, double sigma, bool symmetric) {
    const double x = symmetric ? std::abs(distance) : distance;
    // phi(z)/(sigma * Phi(z)) with z = x / sigma, written through the combined
    // helper so the lower tail stays accurate instead of dividing two small
    // numbers.
    return normalPdfOverCdf(x / sigma) / sigma;
}

bool bandIsResolvable(const TriangleMesh& mesh, const BakeSettings& settings, double& dxOut) {
    const Frame frame = makeFrame(mesh, settings);
    dxOut = frame.dx;
    return frame.dx <= settings.sigma;
}

bool MeshTransform::isIdentity() const {
    return scale == 1.0 && translation == Vector3::Zero();
}

MeshTransform fitMeshTransform(const TriangleMesh& mesh, double maximumExtent, bool center) {
    if (!std::isfinite(maximumExtent)) {
        throw std::invalid_argument("maximum extent must be finite");
    }
    MeshTransform transform;
    // bounds() rejects a degenerate box, so gap is strictly positive and the
    // division below cannot blow up.
    const Bounds3 bounds = mesh.bounds();
    if (center) {
        transform.translation = -0.5 * (bounds.minimum + bounds.maximum);
    }
    if (maximumExtent > 0.0) {
        transform.scale = maximumExtent / (bounds.maximum - bounds.minimum).maxCoeff();
    }
    return transform;
}

void applyMeshTransform(TriangleMesh& mesh, const MeshTransform& transform) {
    if (!std::isfinite(transform.scale) || !(transform.scale > 0.0) ||
        !transform.translation.allFinite()) {
        throw std::invalid_argument("mesh transform must have a finite positive scale");
    }
    if (transform.isIdentity()) return;
    // Checked on the extent rather than on the transformed vertices so the mesh
    // is never left flattened by a transform that throws. A scale tiny enough to
    // underflow every coordinate to zero is the one case the positivity check
    // above cannot see; left alone it would surface later in makeFrame as a
    // degenerate bounding box, far from the transform that caused it. (A merely
    // *small* result is not this function's business: whether the band can be
    // resolved at that size depends on sigma, and bandIsResolvable answers that.)
    const Bounds3 bounds = mesh.bounds();
    if (!(transform.scale * (bounds.maximum - bounds.minimum).minCoeff() > 0.0)) {
        throw std::invalid_argument("mesh transform collapses the mesh onto a plane or point");
    }
    mesh.vertices = (transform.scale * (mesh.vertices.rowwise() + transform.translation.transpose()))
                        .eval();
}

BakedGrids bakeMacrofacetField(const TriangleMesh& mesh, const BakeSettings& settings) {
    validateSettings(settings);
    const Frame frame = makeFrame(mesh, settings);
    const double sigma = settings.sigma;
    const double background = 0.0;

    MeshDistanceField distanceField(mesh, settings.signMode);

    const int nx = frame.hi[0] - frame.lo[0] + 1;
    const int ny = frame.hi[1] - frame.lo[1] + 1;
    const int nz = frame.hi[2] - frame.lo[2] + 1;
    const int workers = resolveWorkerCount(settings.threadCount, nz);

    std::vector<std::vector<BandVoxel>> perWorker(static_cast<std::size_t>(workers));
    std::atomic<int> nextSlab{0};

    auto bakeSlab = [&](int k, std::vector<BandVoxel>& out) {
        const Eigen::Index count = static_cast<Eigen::Index>(nx) * ny;
        Eigen::MatrixXd points(count, 3);
        const int zk = frame.lo[2] + k;
        const double z = frame.dx * zk;
        Eigen::Index row = 0;
        for (int i = 0; i < nx; ++i) {
            const int xi = frame.lo[0] + i;
            const double x = frame.dx * xi;
            for (int j = 0; j < ny; ++j) {
                points(row, 0) = x;
                points(row, 1) = frame.dx * (frame.lo[1] + j);
                points(row, 2) = z;
                ++row;
            }
        }

        Eigen::VectorXd sqrD;
        distanceField.unsignedSquaredDistances(points, sqrD, nullptr);

        // Only points whose unsigned distance could possibly land in the band
        // need a winding number, and the winding number is the expensive half
        // of the bake (roughly 1.6x the distance query, with no batched API to
        // amortize it). |dist| <= sqrt(sqrD) always, so sqrt(sqrD) >= band can
        // only be in the band on an open mesh, where w is fractional.
        const double band = settings.bandSigmas * sigma;
        const double pruneLimit = !settings.fullDomain && settings.pruneByUnsignedDistance
            ? band : std::numeric_limits<double>::infinity();

        row = 0;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j, ++row) {
                const double magnitude = std::sqrt(std::max(0.0, sqrD[row]));
                if (magnitude >= pruneLimit) continue;
                const double winding = distanceField.windingNumber(points.row(row).transpose());
                const double signedDistance = settings.signMode == SignMode::Threshold
                    ? (winding < 0.5 ? magnitude : -magnitude)
                    : magnitude * (1.0 - 2.0 * std::abs(winding));
                if (!settings.fullDomain &&
                    !(signedDistance > -band && signedDistance < band)) continue;
                out.push_back(BandVoxel{frame.lo[0] + i, frame.lo[1] + j, zk,
                                        static_cast<float>(signedDistance),
                                        static_cast<float>(
                                            macrofacetDensity(signedDistance, sigma,
                                                              settings.symmetricDensity))});
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            std::vector<BandVoxel>& out = perWorker[static_cast<std::size_t>(worker)];
            for (;;) {
                const int k = nextSlab.fetch_add(1);
                if (k >= nz) break;
                bakeSlab(k, out);
            }
        });
    }
    for (std::thread& thread : pool) thread.join();

    std::size_t total = 0;
    for (const std::vector<BandVoxel>& chunk : perWorker) total += chunk.size();
    if (total == 0) {
        throw std::invalid_argument(
            "the narrow band contains no voxels: dx is too coarse for this sigma "
            "(increase the resolution or the band width)");
    }

    // --- fill the build grids ---
    // All three grids have the original generator's zero background. setValue
    // retains the same active mask even when an in-band SDF happens to be zero.
    nanovdb::tools::build::Grid<float> sdfGrid(0.0f, "sdf",
                                               nanovdb::GridClass::Unknown);
    nanovdb::tools::build::Grid<float> densityGrid(0.0f, "density",
                                                   nanovdb::GridClass::Unknown);
    nanovdb::tools::build::Grid<float> alphaGrid(0.0f, "alpha",
                                                 nanovdb::GridClass::Unknown);
    sdfGrid.setTransform(frame.dx, nanovdb::Vec3d(frame.origin.x(), frame.origin.y(),
                                                  frame.origin.z()));
    densityGrid.setTransform(frame.dx, nanovdb::Vec3d(frame.origin.x(), frame.origin.y(),
                                                      frame.origin.z()));
    alphaGrid.setTransform(frame.dx, nanovdb::Vec3d(frame.origin.x(), frame.origin.y(),
                                                    frame.origin.z()));

    BakeReport report;
    report.dx = frame.dx;
    report.origin = frame.origin;
    report.domainMinimum = frame.domainMinimum;
    report.domainMaximum = frame.domainMaximum;
    report.voxels = {nx, ny, nz};
    report.bandVoxels = static_cast<long long>(total);
    report.background = background;
    report.threads = workers;
    report.sdfMinimum = std::numeric_limits<double>::infinity();
    report.sdfMaximum = -std::numeric_limits<double>::infinity();
    report.densityMaximum = 0.0;

    const float alphaFloat = static_cast<float>(settings.alpha);
    for (const std::vector<BandVoxel>& chunk : perWorker) {
        for (const BandVoxel& voxel : chunk) {
            const nanovdb::Coord ijk(voxel.i, voxel.j, voxel.k);
            sdfGrid.setValue(ijk, voxel.sdf);
            densityGrid.setValue(ijk, voxel.density);
            alphaGrid.setValue(ijk, alphaFloat);
            report.sdfMinimum = std::min(report.sdfMinimum, static_cast<double>(voxel.sdf));
            report.sdfMaximum = std::max(report.sdfMaximum, static_cast<double>(voxel.sdf));
            report.densityMaximum = std::max(report.densityMaximum, static_cast<double>(voxel.density));
        }
    }

    // Include the background because band cells interpolate toward it. A
    // component bound must be combined across all three gradient components.
    const double rangeMinimum = std::min(report.sdfMinimum, background);
    const double rangeMaximum = std::max(report.sdfMaximum, background);
    report.maximumGradientNorm = detail::trilinearGradientNormBound(
        rangeMinimum, rangeMaximum, frame.dx);

    // --- the constant sigma grid: empty tree, only a root, background carries it ---
    nanovdb::tools::build::Grid<float> sigmaGrid(static_cast<float>(sigma), "sigma",
                                                 nanovdb::GridClass::Unknown);
    sigmaGrid.setTransform(frame.dx, nanovdb::Vec3d(frame.origin.x(), frame.origin.y(),
                                                    frame.origin.z()));

    BakedGrids baked;
    baked.grids.resize(kGridCount);
    baked.grids[kDensityGrid] = nanovdb::tools::createNanoGrid(densityGrid);
    baked.grids[kAlphaGrid] = nanovdb::tools::createNanoGrid(alphaGrid);
    baked.grids[kSdfGrid] = nanovdb::tools::createNanoGrid(sdfGrid);
    baked.grids[kSigmaGrid] = nanovdb::tools::createNanoGrid(sigmaGrid);
    baked.report = report;
    return baked;
}

} // namespace mf

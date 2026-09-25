#pragma once

#include "fieldgen/MeshDistanceField.h"
#include "macrofacet/core/Types.h"
#include "macrofacet/gpss/MeanField.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>

#include <array>
#include <vector>

namespace mf {

struct BakeSettings {
    double sigma = 0.05;   // standard deviation of the surface position
    double alpha = 0.5;    // material NDF parameter written to the alpha grid
    std::array<int, 3> resolution{32, 32, 32};  // voxels per axis over the unexpanded bbox
    double bandSigmas = 3.0;         // the grids are written only inside +-bandSigmas
    double backgroundSigmas = 6.0;   // the sdf grid's background value, in sigma
    int threadCount = 0;             // 0 selects the hardware concurrency
    bool symmetricDensity = false;   // use |distance| in Density() instead of the signed one
    bool fullDomain = false;          // preserve signed distance throughout the baked box
    SignMode signMode = SignMode::Winding;
    // Skips the winding-number query for voxels whose unsigned distance alone
    // rules out the band. Exact for a watertight mesh (where |w| is 0 or 1, so
    // |dist| == sqrt(sqrD)); an open mesh can have a fractional w that shrinks
    // the magnitude into the band, and those voxels would be missed.
    bool pruneByUnsignedDistance = true;
};

struct BakeReport {
    double dx = 0.0;
    Point3 origin = Point3::Zero();
    Point3 domainMinimum = Point3::Zero();
    Point3 domainMaximum = Point3::Zero();
    std::array<long long, 3> voxels{0, 0, 0};  // grid dimensions, per axis
    long long bandVoxels = 0;
    double sdfMinimum = 0.0;
    double sdfMaximum = 0.0;
    double background = 0.0;
    double densityMaximum = 0.0;
    // A provable upper bound on |grad m| for the isotropic trilinear grid:
    // sqrt(3) * (max value - min value) / dx. It is only ever an upper bound, so an
    // over-estimate loosens the majorant (slower tracking) but never drops a
    // collision. See the note in VdbBaker.cpp.
    double maximumGradientNorm = 0.0;
    int threads = 1;
};

// The four grids live in one vector rather than as separate members because
// nanovdb::io::writeGrids takes the whole set as a single const container, and
// GridHandle is move-only: they cannot be gathered into a container afterwards.
enum GridIndex {
    kDensityGrid = 0,
    kAlphaGrid = 1,
    kSdfGrid = 2,
    kSigmaGrid = 3,
    kGridCount = 4,
};

struct BakedGrids {
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> grids;  // in GridIndex order
    BakeReport report;

    const nanovdb::GridHandle<nanovdb::HostBuffer>& density() const { return grids[kDensityGrid]; }
    const nanovdb::GridHandle<nanovdb::HostBuffer>& alpha() const { return grids[kAlphaGrid]; }
    const nanovdb::GridHandle<nanovdb::HostBuffer>& sdf() const { return grids[kSdfGrid]; }
    const nanovdb::GridHandle<nanovdb::HostBuffer>& sigma() const { return grids[kSigmaGrid]; }
};

// Density(x, sigma, k) = k * phi(x; 0, sigma) / Phi(x / sigma): the macrofacet
// edge density, i.e. the hazard of the underlying Gaussian surface model.
// `symmetric` evaluates it at |distance| instead, which the original generator
// did not do -- signed distances below zero inflate the density by up to 4x.
double macrofacetDensity(double distance, double sigma, bool symmetric);

// Bakes the three grids (sdf / density / alpha) plus the constant sigma grid.
// Throws std::invalid_argument for settings that cannot produce a usable field
// (non-positive sigma, empty mesh, a grid too coarse to contain the band).
BakedGrids bakeMacrofacetField(const TriangleMesh& mesh, const BakeSettings& settings);

// Samples a procedural mean throughout the active transport domain, including
// one interpolation-cell halo. Unlike the mesh baker this preserves the signed
// field in the interior instead of replacing it with the exterior background.
BakedGrids bakeAnalyticMean(const MeanField& mean, const Bounds3& domain,
                           double voxelSize, double sigma, double alpha);

// True when dx is too coarse for the band to contain any voxel; the caller uses
// this to warn and suggest a resolution rather than emitting an empty grid.
bool bandIsResolvable(const TriangleMesh& mesh, const BakeSettings& settings, double& dxOut);

// A uniform transform applied to a mesh before baking. sigma is a world-space
// length, so a mesh authored in arbitrary units (every imported model is) has
// to be brought to the scale sigma is expressed in -- otherwise the band is
// either invisible or swallows the whole object. Nothing inside the .nvdb
// records what the source mesh's units were, so this is written to the sidecar:
// without it a field cannot be reproduced.
//
// The point mapping is p' = scale * (p + translation): translation first, so
// the scale is about the translated origin.
struct MeshTransform {
    double scale = 1.0;
    Vector3 translation = Vector3::Zero();

    bool isIdentity() const;
};

// The transform that maps the mesh's bounding box to one centred on the origin
// whose largest extent is `maximumExtent`. A non-positive `maximumExtent` leaves
// the size alone, so it degrades to a pure centring -- and with `center` false,
// to the identity. `center` false leaves the translation zero either way, so the
// scale stays about the mesh's own origin.
MeshTransform fitMeshTransform(const TriangleMesh& mesh, double maximumExtent, bool center);

// Applies `transform` to mesh.vertices in place. Throws std::invalid_argument if
// the transform is not finite, or if it collapses the mesh onto a plane or point.
void applyMeshTransform(TriangleMesh& mesh, const MeshTransform& transform);

} // namespace mf

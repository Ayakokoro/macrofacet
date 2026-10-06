#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/gpss/MeanField.h"

#include <filesystem>
#include <memory>

namespace mf {

// A mean field backed by the `sdf` grid of a .nvdb written by
// macrofacet_fieldgen. It is an ordinary MeanField, so the renderer treats a
// baked field and an analytic one (SphereMean, ShaderBallMean, ...) identically.
//
// Two behaviours differ from the analytic fields on purpose:
//
//  - Where all eight interpolation samples are inactive, evaluate() returns
//    the grid's background (zero for mesh bakes) with zero gradient. At active/inactive
//    boundaries it blends stored values with the background and can have a
//    nonzero gradient. Queries outside the grid remain defined.
//  - bounds() uses the grid's global value range and gradient bound for every
//    domain. This is conservative for sub-domains but may loosen a majorant.
//
// This header deliberately exposes no nanovdb type; the grid lives behind a
// pimpl so the core library never sees the dependency.
class NanoVdbMean final : public MeanField {
public:
    // `sigmaOverride` <= 0 takes sigma from the .nvdb (its constant "sigma"
    // grid), falling back to the sidecar. A positive value does not replace it:
    // the file is the source of truth for a baked field, so the value is only
    // cross-checked against the file, and a disagreement throws.
    static std::shared_ptr<const NanoVdbMean> open(const std::filesystem::path& gridFile,
                                                   double sigmaOverride = 0.0);

    const char* typeName() const override { return "nanovdb"; }
    // The bake fixed sigma, so the file is this field's only valid source for
    // it -- see MeanField::intrinsicSigma.
    std::optional<double> intrinsicSigma() const override;
    MeanJet evaluate(const Point3& x) const override;
    double valueDifference(const Point3& x, const Vector3& displacement) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    MeanRayBounds rayBounds(const Point3& origin, const Vector3& direction,
                            double begin, double end) const override;
    double voxelSizeHint() const override;
    void appendRayBreakpoints(const Point3& origin, const Vector3& direction,
                              double begin, double end,
                              std::vector<double>& knots) const override;
    void requireFullRayCoverage(const Point3& origin, const Vector3& direction,
                                double maximumDistance) const override;

    ~NanoVdbMean() override;

    double sigma() const;
    double background() const;
    const std::filesystem::path& gridFile() const;
    // Includes the interpolation cell after the highest active node.
    const Bounds3& gridBounds() const;
    // Bounding box through the outermost stored SDF nodes; not the exact
    // active-band mask of a sparse bake.
    const Bounds3& activeNodeBounds() const;

private:
    NanoVdbMean();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Registers "nanovdb" with the mean-field factory. Call it once at startup from
// any executable that links macrofacet_field; it is idempotent. This is an
// explicit call rather than a static initializer because a static library's
// registration object is dropped by the linker when nothing references it.
void registerNanoVdbFieldTypes();

} // namespace mf

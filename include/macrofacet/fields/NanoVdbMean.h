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
//  - Outside the grid, and at inactive voxels inside it, evaluate() returns the
//    grid's background (+6 sigma) with a zero gradient. It must not throw:
//    The flight kernel evaluates the mean at the ray's birth point during
//    construction, and that point lies on the config's active domain, which is
//    larger than the baked grid.
//  - bounds() reports the grid's global extremes for every domain. That is
//    conservative in both directions (a sub-domain is a subset), so it can only
//    loosen the majorant, never drop a collision.
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
    BoundsSummary bounds(const Bounds3& domain) const override;
    double voxelSizeHint() const override;
    void appendRayBreakpoints(const Point3& origin, const Vector3& direction,
                              double begin, double end,
                              std::vector<double>& knots) const override;

    ~NanoVdbMean() override;

    double sigma() const;
    double background() const;
    const std::filesystem::path& gridFile() const;
    // The grid's world-space bounding box, which is what the bake covered.
    const Bounds3& gridBounds() const;

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

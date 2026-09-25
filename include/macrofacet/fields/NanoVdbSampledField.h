#pragma once

#include "macrofacet/fields/ScalarField.h"

#include <filesystem>
#include <memory>
#include <string>

namespace mf {

// The `alpha` grid of a baked .nvdb, exposed as a per-point scalar field. It
// drives the material NDF (per-point roughness), never the transport
// statistics -- see docs/archive/PLAN_NANOVDB_FIELD.md 1.3.
//
// Sampling is trilinear. Outside the grid the grid's background is returned,
// which the generator sets to the same constant alpha it writes inside the
// band, so the field does not jump at the band edge.
class NanoVdbSampledField final : public ScalarField {
public:
    static std::shared_ptr<const NanoVdbSampledField> open(const std::filesystem::path& gridFile,
                                                           const std::string& gridName = "alpha");

    double sample(const Point3& x) const override;
    ScalarBounds bounds(const Bounds3& domain) const override;

    ~NanoVdbSampledField() override;

    double minimumValue() const;
    double maximumValue() const;
    double background() const;
    const std::string& gridName() const;
    const Bounds3& gridBounds() const;
    double voxelSize() const;

private:
    NanoVdbSampledField();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mf

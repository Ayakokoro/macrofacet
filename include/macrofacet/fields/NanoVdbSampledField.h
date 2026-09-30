#pragma once

#include "macrofacet/fields/ScalarField.h"

#include <filesystem>
#include <memory>
#include <string>

namespace mf {

// The `alpha` grid of a baked .nvdb, exposed as a per-point scalar field. It
// drives the local material NDF and its projected area in transport.
//
// Sampling is trilinear. Mesh bakes use a zero background for density and alpha;
// analytic bakes keep a positive alpha background.
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
    bool positiveAtEveryPositiveNodeOf(const NanoVdbSampledField& density) const;

private:
    NanoVdbSampledField();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mf

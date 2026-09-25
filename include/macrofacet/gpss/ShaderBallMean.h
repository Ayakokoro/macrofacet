#pragma once

#include "macrofacet/gpss/MeanField.h"
#include <array>

namespace mf {

// A sphere with two circular toroidal grooves and a rounded circular pedestal.
// The grooves rotate with grooveAxis; the pedestal always stands along world Z.
// evaluate() returns the exact signed Euclidean distance, including at the
// groove rims. At medial axes and sharp seams its unit gradient is a
// deterministic one-sided choice.
class ShaderBallMean final : public MeanField {
public:
    const char* typeName() const override { return "shader_ball"; }
    ShaderBallMean(Point3 center, double radius, Vector3 grooveAxis, double grooveRadius);
    MeanJet evaluate(const Point3& x) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;

    const Point3& center() const { return center_; }
    double radius() const { return radius_; }
    const Vector3& grooveAxis() const { return grooveAxis_; }
    double grooveRadius() const { return grooveRadius_; }

private:
    struct Arc {
        Vector2 center = Vector2::Zero();
        double radius = 0.0;
        double normalSign = 1.0;
        Vector2 firstDirection = Vector2::Zero();
        Vector2 lastDirection = Vector2::Zero();
        Vector2 first = Vector2::Zero();
        Vector2 last = Vector2::Zero();
    };

    MeanJet ballJet(const Vector3& p) const;
    MeanJet pedestalJet(const Vector3& p) const;

    Point3 center_;
    double radius_;
    Vector3 grooveAxis_;
    double grooveRadius_;
    Vector3 radialFallback_;
    double grooveMajorRadius_;
    std::array<Arc, 5> arcs_;
};

} // namespace mf

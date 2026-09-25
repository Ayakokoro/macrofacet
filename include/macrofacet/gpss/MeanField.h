#pragma once

#include "macrofacet/core/Types.h"
#include <memory>
#include <optional>
#include <vector>

namespace mf {

struct MeanJet {
    double value = 0.0;
    Vector3 gradient = Vector3::Zero();
};

struct BoundsSummary {
    double minimumValue = 0.0;
    double maximumGradientNorm = 0.0;
    bool certified = false;
};

class MeanField {
public:
    virtual ~MeanField() = default;
    // The config's "mean_type" string. This is what lets a resolved config echo
    // itself back out through the same registry that built it, so the core
    // library needs no dynamic_cast per field type -- and a type it has never
    // heard of, such as a baked NanoVDB grid, still round-trips.
    virtual const char* typeName() const = 0;
    // The statistical scale the field fixes by itself, when it has one. A
    // procedural mean is scale-free -- the same sphere renders at any sigma, so
    // sigma there is a model knob and the config is its only source. A *baked*
    // field is not: sigma is frozen into its data, since the band spans +-3
    // sigma, the background sits at +6 sigma and dx was chosen against it. The
    // config may then omit "sigma" and read it from here, which is why this is
    // a capability of the field rather than a field of its own: nothing
    // downstream can tell the two sources apart, because kernel.sigma() is
    // derived from the resolved value either way.
    virtual std::optional<double> intrinsicSigma() const { return std::nullopt; }
    virtual MeanJet evaluate(const Point3& x) const = 0;
    virtual double valueDifference(const Point3& x, const Vector3& displacement) const {
        if (const auto gradient = affineGradient()) return gradient->dot(displacement);
        return evaluate(x + displacement).value - evaluate(x).value;
    }
    virtual BoundsSummary bounds(const Bounds3& domain) const = 0;
    virtual std::optional<Vector3> affineGradient() const { return std::nullopt; }
    // Nonzero for a sampled grid. The integrator partitions at interpolation
    // cell boundaries rather than using a loose global gradient bound.
    virtual double voxelSizeHint() const { return 0.0; }
    virtual void appendRayBreakpoints(const Point3&, const Vector3&, double, double,
                                      std::vector<double>&) const {}
};

class PlaneMean final : public MeanField {
public:
    const char* typeName() const override { return "plane"; }
    PlaneMean(Vector3 normal, double offset);
    MeanJet evaluate(const Point3& x) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    std::optional<Vector3> affineGradient() const override { return normal_; }
    const Vector3& normal() const { return normal_; }
    double offset() const { return offset_; }
private:
    Vector3 normal_;
    double offset_;
};

class SphereMean final : public MeanField {
public:
    const char* typeName() const override { return "sphere"; }
    SphereMean(Point3 center, double radius);
    MeanJet evaluate(const Point3& x) const override;
    double valueDifference(const Point3& x, const Vector3& displacement) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
private:
    Point3 center_;
    double radius_;
};

// A solid sphere with the quarter-sector x > 0, y > 0 removed outside
// innerRadius. Coordinates are relative to center; the inner sphere is kept.
// evaluate() returns the exact signed Euclidean distance. At sharp edges or
// medial-axis points its unit gradient is a deterministic one-sided choice.
class CutawaySphereMean final : public MeanField {
public:
    const char* typeName() const override { return "cutaway_sphere"; }
    CutawaySphereMean(Point3 center, double innerRadius, double outerRadius);
    MeanJet evaluate(const Point3& x) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    const Point3& center() const { return center_; }
    double innerRadius() const { return innerRadius_; }
    double outerRadius() const { return outerRadius_; }
private:
    Point3 center_;
    double innerRadius_;
    double outerRadius_;
};

class ConstantMean final : public MeanField {
public:
    const char* typeName() const override { return "constant"; }
    explicit ConstantMean(double value) : value_(value) {}
    MeanJet evaluate(const Point3&) const override { return {value_, Vector3::Zero()}; }
    BoundsSummary bounds(const Bounds3&) const override { return {value_, 0.0, true}; }
    std::optional<Vector3> affineGradient() const override { return Vector3::Zero(); }
private:
    double value_;
};

using MeanFieldPtr = std::shared_ptr<const MeanField>;

} // namespace mf

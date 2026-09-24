#pragma once

#include "macrofacet/core/Types.h"
#include <memory>
#include <optional>

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
    virtual MeanJet evaluate(const Point3& x) const = 0;
    virtual BoundsSummary bounds(const Bounds3& domain) const = 0;
    virtual std::optional<Vector3> affineGradient() const { return std::nullopt; }
};

class PlaneMean final : public MeanField {
public:
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
    SphereMean(Point3 center, double radius);
    MeanJet evaluate(const Point3& x) const override;
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
    explicit ConstantMean(double value) : value_(value) {}
    MeanJet evaluate(const Point3&) const override { return {value_, Vector3::Zero()}; }
    BoundsSummary bounds(const Bounds3&) const override { return {value_, 0.0, true}; }
    std::optional<Vector3> affineGradient() const override { return Vector3::Zero(); }
private:
    double value_;
};

using MeanFieldPtr = std::shared_ptr<const MeanField>;

} // namespace mf


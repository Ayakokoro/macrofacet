#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/mathutility/SmallGaussian.h"
#include <vector>

namespace mf {

class ConditionedRay {
public:
    ConditionedRay(const GPSSField& field, Point3 x0, Vector3 g0, Vector3 w,
                   const NumericPolicy& policy = defaultNumericPolicy());

    Gaussian<2> endpointValueSlope(double t) const;
    Gaussian<3> midpointValueSlope(double t) const;
    Gaussian<4> endpointValueGradient(double t) const;
    Gaussian<5> midpointValueGradient(double t) const;
    DynamicGaussian valuesAt(const std::vector<double>& ages) const;
    DynamicGaussian checkpointValuesAndEndpointSlope(
        const std::vector<double>& interiorAges, double t) const;

    const Point3& origin() const { return x0_; }
    const Vector3& birthGradient() const { return g0_; }
    const Vector3& direction() const { return w_; }
    const GPSSField& field() const { return field_; }

private:
    struct Descriptor { Point3 point; int component; }; // -1 value, 0..2 gradient

    DynamicGaussian conditionDescriptors(const std::vector<Descriptor>& descriptors) const;
    double priorCovariance(const Descriptor& a, const Descriptor& b) const;
    double stableValueCovariance(double s, double t) const;
    double stableValueSlopeCovariance(double s, double t) const;
    double stableSlopeVariance(double t) const;
    std::pair<double, double> conditionedValueSlopeMean(double t) const;

    const GPSSField& field_;
    Point3 x0_;
    Vector3 g0_;
    Vector3 w_;
    Gaussian<4> observation_;
    Eigen::Vector4d observed_;
    double rayPrecision_;
    NumericPolicy policy_;
};

} // namespace mf


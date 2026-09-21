#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <cmath>
#include <limits>

namespace mf {

ClassicEvaluation evaluateClassic(const GPSSField& field, const Point3& x, const Vector3& w,
                                  const NumericPolicy& policy) {
    const PointPrior prior = field.pointPrior(x);
    const double sigma = field.kernel.sigma();
    const double z = prior.meanF / sigma;
    const PositiveResult density = positiveFromLog(
        normalLogPdf(z) - std::log(sigma) - normalLogCdf(z));
    PositiveResult area;
    if (field.ndfFamily == NdfFamily::GGXBaseline) {
        area = GgxHeightfield(field.ggxAlpha.x(), field.ggxAlpha.y()).projectedArea(w);
    } else {
        const Vector3 direction = normalizedOrThrow(w);
        const double meanProjection = direction.dot(prior.meanG);
        const double varianceProjection = validateNonnegative(
            direction.dot(prior.covarianceG * direction), prior.covarianceG.norm(), policy);
        area = negativePartMean(meanProjection, std::sqrt(varianceProjection), policy);
    }
    PositiveResult extinction;
    if (density.status == NumericStatus::ExactZero || area.status == NumericStatus::ExactZero) {
        extinction = exactZero();
    } else {
        extinction = positiveFromLog(density.logValue + area.logValue,
                                     density.absError * area.value + area.absError * density.value);
    }
    return {density, area, extinction};
}

double classicMajorant(const GPSSField& field, const Bounds3& domain, const Vector3& w,
                       const NumericPolicy& policy) {
    (void)policy;
    const BoundsSummary bounds = field.mean->bounds(domain);
    if (!bounds.certified) {
        throw NumericError(NumericStatus::InvalidInput, "mean field has no certified bounds");
    }
    const double sigma = field.kernel.sigma();
    const double z = bounds.minimumValue / sigma;
    const double rhoMaximum = std::exp(normalLogPdf(z) - std::log(sigma) - normalLogCdf(z));
    double areaMaximum = 0.0;
    if (field.ndfFamily == NdfFamily::GGXBaseline) {
        const Vector3 wn = normalizedOrThrow(w);
        areaMaximum = 0.5 * (std::sqrt(wn.z() * wn.z() +
            field.ggxAlpha.x() * field.ggxAlpha.x() * wn.x() * wn.x() +
            field.ggxAlpha.y() * field.ggxAlpha.y() * wn.y() * wn.y()) - wn.z());
    } else {
        const Matrix3 covariance = sigma * sigma * field.kernel.precision();
        areaMaximum = std::sqrt(bounds.maximumGradientNorm * bounds.maximumGradientNorm +
                                covariance.trace());
    }
    const double value = rhoMaximum * areaMaximum;
    return std::nextafter(value, std::numeric_limits<double>::infinity());
}

} // namespace mf

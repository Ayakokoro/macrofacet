#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {
namespace {

PositiveResult projectedArea(const PointPrior& prior, const MaterialConfig& material,
                             const Point3& x, const Vector3& w,
                             const NumericPolicy& policy) {
    if (material.ndfFamily == NdfFamily::GGXBaseline) {
        const Vector2 alpha = material.ggxAlphaAt(x);
        return GgxHeightfield(alpha.x(), alpha.y()).projectedArea(w);
    }
    const Vector3 direction = normalizedOrThrow(w);
    const double meanProjection = direction.dot(prior.meanG);
    const double varianceProjection = validateNonnegative(
        direction.dot(prior.covarianceG * direction), prior.covarianceG.norm(), policy);
    return negativePartMean(meanProjection, std::sqrt(varianceProjection), policy);
}

} // namespace

PositiveResult classicProjectedArea(const GPSSField& field, const MaterialConfig& material,
                                    const Point3& x, const Vector3& w,
                                    const NumericPolicy& policy) {
    return projectedArea(field.pointPrior(x), material, x, w, policy);
}

ClassicEvaluation evaluateClassic(const GPSSField& field, const MaterialConfig& material,
                                  const Point3& x, const Vector3& w,
                                  const NumericPolicy& policy) {
    // Role 1 throughout. The projected area below is the transport quantity --
    // the collision-gradient distribution -- not the shading NDF, so it must not
    // read the alpha grid. Alpha reaches the extinction only through the GGX
    // branch (docs/archive/PLAN_NANOVDB_FIELD.md 1.4 A/D).
    const PointPrior prior = field.pointPrior(x);
    const double sigma = field.kernel.sigma();
    const double z = prior.meanF / sigma;
    const PositiveResult density = positiveFromLog(
        normalLogPdf(z) - std::log(sigma) - normalLogCdf(z));
    const PositiveResult area = projectedArea(prior, material, x, w, policy);
    PositiveResult extinction;
    if (density.status == NumericStatus::ExactZero || area.status == NumericStatus::ExactZero) {
        extinction = exactZero();
    } else {
        extinction = positiveFromLog(density.logValue + area.logValue,
                                     density.absError * area.value + area.absError * density.value);
    }
    return {density, area, extinction};
}

double classicAreaMajorant(const GPSSField& field, const MaterialConfig& material,
                           const Bounds3& domain, const Vector3& w,
                           const NumericPolicy& policy) {
    (void)policy;
    const BoundsSummary bounds = field.mean->bounds(domain);
    if (!bounds.certified) {
        throw NumericError(NumericStatus::InvalidInput, "mean field has no certified bounds");
    }
    const double sigma = field.kernel.sigma();
    double areaMaximum = 0.0;
    if (material.ndfFamily == NdfFamily::GGXBaseline) {
        // GGX is the only family whose extinction reads the alpha grid, and it
        // reads it per point via ggxAlphaAt(). The projected-area formula below
        // increases with alpha, so the domain needs the *maximum* alpha: a
        // majorant built from the config's ggxAlpha goes below the true value
        // wherever alpha(x) exceeds it, and the tracker then drops real
        // collisions. This is the one place in the whole integration where
        // getting the bound wrong changes the rendered result.
        Vector2 alpha = material.ggxAlpha;
        if (material.alphaField) {
            const ScalarBounds alphaBounds = material.alphaField->bounds(domain);
            if (!alphaBounds.certified) {
                throw NumericError(NumericStatus::InvalidInput,
                                   "alpha field has no certified bounds");
            }
            // Scalar lifted to isotropic, matching ggxAlphaAt().
            alpha = Vector2::Constant(std::max(0.0, alphaBounds.maximumValue));
        }
        const Vector3 wn = normalizedOrThrow(w);
        areaMaximum = 0.5 * (std::sqrt(wn.z() * wn.z() +
            alpha.x() * alpha.x() * wn.x() * wn.x() +
            alpha.y() * alpha.y() * wn.y() * wn.y()) - wn.z());
    } else {
        // The Gaussian family's projected area is role 1 and reads no alpha, so
        // the constant covariance is still the right bound here.
        const Matrix3 covariance = sigma * sigma * field.kernel.precision();
        areaMaximum = std::sqrt(bounds.maximumGradientNorm * bounds.maximumGradientNorm +
                                covariance.trace());
    }
    return std::nextafter(areaMaximum, std::numeric_limits<double>::infinity());
}

double classicMajorant(const GPSSField& field, const MaterialConfig& material,
                       const Bounds3& domain, const Vector3& w,
                       const NumericPolicy& policy) {
    const BoundsSummary bounds = field.mean->bounds(domain);
    if (!bounds.certified) {
        throw NumericError(NumericStatus::InvalidInput, "mean field has no certified bounds");
    }
    const double sigma = field.kernel.sigma();
    const double z = bounds.minimumValue / sigma;
    const double rhoMaximum = std::exp(normalLogPdf(z) - std::log(sigma) - normalLogCdf(z));
    const double value = rhoMaximum * classicAreaMajorant(field, material, domain, w, policy);
    return std::nextafter(value, std::numeric_limits<double>::infinity());
}

} // namespace mf

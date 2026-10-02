#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/macrofacet/LocalFrame.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {
namespace {

PositiveResult projectedArea(const PointPrior& prior, const Vector3& w,
                             const NumericPolicy& policy) {
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
    if (material.ndfFamily == NdfFamily::GGXBaseline) {
        const Vector2 alpha = material.ggxAlphaAt(x);
        const Vector3 normal = field.mean->evaluate(x).gradient;
        return GgxHeightfield(alpha.x(), alpha.y()).projectedArea(
            tangentFrame(normal).transpose() * w);
    }
    return projectedArea(material.materialNdf(field, x), w, policy);
}

ClassicEvaluation evaluateClassic(const GPSSField& field, const MaterialConfig& material,
                                  const Point3& x, const Vector3& w,
                                  const NumericPolicy& policy) {
    // Both GP models use the same one-point height density. The selected
    // gradient distribution is shared by extinction and collision scattering.
    const double distance = field.mean->evaluate(x).value;
    const double sigma = field.kernel.sigma();
    const double z = distance / sigma;
    const PositiveResult density = positiveFromLog(
        normalLogPdf(z) - std::log(sigma) - normalLogCdf(z));
    const PositiveResult area = classicProjectedArea(field, material, x, w, policy);
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
    (void)w;
    double areaMaximum = 0.0;
    if (material.ndfFamily == NdfFamily::GGXBaseline) {
        // Bound the GGX projected area for every tangent-frame orientation.
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
        areaMaximum = 0.5 * (1.0 + std::max({1.0, alpha.x(), alpha.y()}));
    } else {
        // E[(-w.G)+] <= E[||G||] <= sqrt(E[||G||^2]). Global mode
        // retains the mean SDF gradient; local mode normalizes it.
        double maximumMeanGradient = 1.0;
        if (material.gpModel == GpModel::GlobalPointwise) {
            const BoundsSummary bounds = field.mean->bounds(domain);
            if (!bounds.certified || !(bounds.maximumGradientNorm >= 0.0) ||
                !std::isfinite(bounds.maximumGradientNorm)) {
                throw NumericError(NumericStatus::InvalidMajorant,
                                   "global mean has no certified gradient bound");
            }
            maximumMeanGradient = bounds.maximumGradientNorm;
        }
        double trace = field.kernel.gradientCovarianceAtZero().trace();
        if (material.alphaField) {
            const ScalarBounds alphaBounds = material.alphaField->bounds(domain);
            if (!alphaBounds.certified) {
                throw NumericError(NumericStatus::InvalidInput,
                                   "alpha field has no certified bounds");
            }
            trace = 1.5 * alphaBounds.maximumValue * alphaBounds.maximumValue;
        }
        areaMaximum = std::hypot(maximumMeanGradient, std::sqrt(trace));
    }
    return std::nextafter(areaMaximum, std::numeric_limits<double>::infinity());
}

} // namespace mf

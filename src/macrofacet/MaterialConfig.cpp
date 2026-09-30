#include "macrofacet/macrofacet/MaterialConfig.h"
#include <stdexcept>

namespace mf {

PointPrior MaterialConfig::materialNdf(const GPSSField& field, const Point3& x) const {
    PointPrior prior = field.pointPrior(x);
    if (gpModel == GpModel::GlobalPointwise) return prior;
    // The material GP is centred on the tangent plane at x. Its local mean is
    // m_local(u) = u_z, so at the local origin meanF = 0 and meanG = e_z.
    // Convert that unit gradient back to world space. A constant mean has no
    // tangent frame and retains its zero-gradient isotropic-medium model.
    const double meanGradientLength = prior.meanG.norm();
    if (meanGradientLength > 0.0) {
        prior.meanF = 0.0;
        prior.meanG /= meanGradientLength;
    }
    if (alphaField) {
        const double alpha = alphaField->sample(x);
        // The local isotropic SE model has ell = sqrt(2) * sigma / alpha.
        // Its gradient covariance, including the local normal component, is
        // sigma^2 / ell^2 I = alpha^2 / 2 I. Without a grid, the configured
        // global roughness supplies this same covariance through field.kernel.
        prior.covarianceG = (0.5 * alpha * alpha) * Matrix3::Identity();
    }
    return prior;
}

Vector2 MaterialConfig::ggxAlphaAt(const Point3& x) const {
    if (!alphaField) return ggxAlpha;
    const double alpha = alphaField->sample(x);
    return Vector2(alpha, alpha);
}

void MaterialConfig::validate(const Bounds3& domain) const {
    if (gpModel == GpModel::GlobalPointwise &&
        (alphaField || ndfFamily != NdfFamily::GeneralizedGaussian)) {
        throw std::invalid_argument(
            "global_pointwise requires generalized_gaussian and no alpha grid");
    }
    if ((conductor.eta.array() < 0.0).any() || (conductor.k.array() < 0.0).any() ||
        !conductor.eta.allFinite() || !conductor.k.allFinite() ||
        ((conductor.eta.array().square() + conductor.k.array().square()) <= 0.0).any()) {
        throw std::invalid_argument("invalid conductor optical constants");
    }
    if (ndfFamily == NdfFamily::GGXBaseline &&
        (!(ggxAlpha.x() > 0.0) || !(ggxAlpha.y() > 0.0))) {
        throw std::invalid_argument("GGX alpha must be positive");
    }
    if (alphaField) {
        const ScalarBounds bounds = alphaField->bounds(domain);
        if (!bounds.certified) {
            throw std::invalid_argument("the alpha field cannot certify its bounds");
        }
        if (!(bounds.minimumValue > 0.0) &&
            !(gpModel == GpModel::LocalTangent && alphaPositiveOnDensitySupport &&
              bounds.minimumValue == 0.0)) {
            throw std::invalid_argument(
                "the alpha field must be positive wherever density can be positive "
                "(minimum is " + std::to_string(bounds.minimumValue) + ")");
        }
    }
}

} // namespace mf

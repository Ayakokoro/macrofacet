#include "macrofacet/macrofacet/MaterialConfig.h"
#include <stdexcept>

namespace mf {

PointPrior MaterialConfig::materialNdf(const GPSSField& field, const Point3& x) const {
    PointPrior prior = field.pointPrior(x);
    if (alphaField) {
        const double alpha = alphaField->sample(x);
        prior.covarianceG = (alpha * alpha) * Matrix3::Identity();
    }
    return prior;
}

Vector2 MaterialConfig::ggxAlphaAt(const Point3& x) const {
    if (!alphaField) return ggxAlpha;
    const double alpha = alphaField->sample(x);
    return Vector2(alpha, alpha);
}

void MaterialConfig::validate(const Bounds3& domain) const {
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
        if (!(bounds.minimumValue > 0.0)) {
            throw std::invalid_argument(
                "the alpha field must be strictly positive over the active domain "
                "(minimum is " + std::to_string(bounds.minimumValue) + ")");
        }
    }
}

} // namespace mf

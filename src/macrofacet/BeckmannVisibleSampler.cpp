// Adapted from pbrt-v3 src/core/microfacet.cpp and src/core/pbrt.h.
// Copyright (c) 1998-2016 Matt Pharr, Greg Humphreys, and Wenzel Jakob.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "macrofacet/macrofacet/BeckmannVisibleSampler.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mf {
namespace {

void validateAlpha(double alphaX, double alphaY) {
    if (!(alphaX > 0.0) || !(alphaY > 0.0) ||
        !std::isfinite(alphaX) || !std::isfinite(alphaY)) {
        throw std::invalid_argument("Beckmann roughness must be finite and positive");
    }
}

double inverseErf(double x) {
    x = std::clamp(x, std::nextafter(-1.0, 0.0), std::nextafter(1.0, 0.0));
    // PBRT's polynomial seed, followed by two Newton steps against std::erf.
    // Its original 0.99999 clamp drops the tails; use the existing normal
    // quantile there so the sampled support still matches the analytic PDF.
    if (std::abs(x) > 0.99999) {
        const double tail = 0.5 * (1.0 - std::abs(x));
        return std::copysign(-normalQuantile(tail) / kSqrtTwo, x);
    }
    double w = -std::log((1.0 - x) * (1.0 + x));
    double p;
    if (w < 5.0) {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = std::sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    double value = p * x;
    for (int i = 0; i < 2; ++i) {
        value -= (std::erf(value) - x) /
                 (2.0 / std::sqrt(kPi) * std::exp(-value * value));
    }
    return value;
}

Vector2 sampleUnitBeckmannSlopes(double cosTheta, double u1, double u2) {
    if (cosTheta > 0.9999) {
        const double radius = std::sqrt(-std::log1p(-u1));
        const double phi = 2.0 * kPi * u2;
        return Vector2(radius * std::cos(phi), radius * std::sin(phi));
    }

    // PBRT's numerical inversion of the visible slope CDF in erf space.
    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    const double tanTheta = sinTheta / cosTheta;
    const double cotTheta = cosTheta / sinTheta;
    double lo = -1.0;
    double hi = std::erf(cotTheta);
    const double theta = std::acos(cosTheta);
    const double fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594 * theta));
    double b = hi - (1.0 + hi) * std::pow(1.0 - u1, fit);
    const double invSqrtPi = 1.0 / std::sqrt(kPi);
    const double normalization = 1.0 /
        (1.0 + hi + invSqrtPi * tanTheta * std::exp(-cotTheta * cotTheta));
    for (int i = 0; i < 9; ++i) {
        if (!(b > lo && b < hi)) b = 0.5 * (lo + hi);
        const double slopeX = inverseErf(b);
        const double cdf = normalization *
            (1.0 + b + invSqrtPi * tanTheta * std::exp(-slopeX * slopeX));
        const double error = cdf - u1;
        if (std::abs(error) < 1e-12) break;
        if (error > 0.0) hi = b;
        else lo = b;
        const double derivative = normalization * (1.0 - slopeX * tanTheta);
        b -= error / derivative;
    }
    if (!(b > lo && b < hi)) b = 0.5 * (lo + hi);
    return Vector2(inverseErf(b), inverseErf(2.0 * u2 - 1.0));
}

} // namespace

Vector3 samplePbrtBeckmannVisible(const Vector3& travelDirection, double alphaX,
                                 double alphaY, Random& rng) {
    validateAlpha(alphaX, alphaY);
    const Vector3 view = -normalizedOrThrow(travelDirection);
    const double sign = view.z() < 0.0 ? -1.0 : 1.0;
    Vector3 stretched(sign * alphaX * view.x(), sign * alphaY * view.y(),
                      sign * view.z());
    // The visible-slope CDF has a well-defined grazing limit. A tiny positive
    // cosine avoids division by zero at exactly tangent incidence.
    stretched.z() = std::max(stretched.z(), 1e-12);
    stretched.normalize();
    const Vector2 slopes = sampleUnitBeckmannSlopes(stretched.z(), rng.openUniform01(),
                                                    rng.openUniform01());
    const double radial = std::hypot(stretched.x(), stretched.y());
    const double cosPhi = radial > 0.0 ? stretched.x() / radial : 1.0;
    const double sinPhi = radial > 0.0 ? stretched.y() / radial : 0.0;
    const double slopeX = alphaX * (cosPhi * slopes.x() - sinPhi * slopes.y());
    const double slopeY = alphaY * (sinPhi * slopes.x() + cosPhi * slopes.y());
    return sign * normalizedOrThrow(Vector3(-slopeX, -slopeY, 1.0));
}

double pbrtBeckmannVisiblePdf(const Vector3& normal, const Vector3& travelDirection,
                             double alphaX, double alphaY) {
    validateAlpha(alphaX, alphaY);
    const Vector3 w = normalizedOrThrow(travelDirection);
    const double sign = w.z() > 0.0 ? -1.0 : 1.0;
    const GaussianNdf beckmann(Vector3::UnitZ(),
        (Vector3(0.5 * alphaX * alphaX, 0.5 * alphaY * alphaY, 0.0)).asDiagonal());
    return beckmann.visibleNormalPdf(sign * normal, sign * w);
}

LocalBeckmannVisibleSampler::LocalBeckmannVisibleSampler(
    const Vector3& meanGradient, const Matrix3& covarianceGradient) {
    if (!meanGradient.allFinite() || !covarianceGradient.allFinite()) {
        throw std::invalid_argument("invalid local Beckmann gradient statistics");
    }
    // A zero mean gradient has no unique tangent plane. Keep a deterministic
    // frame there; the uniform component retains full proposal support.
    const double meanLength = meanGradient.norm();
    if (!std::isfinite(meanLength)) {
        throw std::invalid_argument("invalid local Beckmann mean gradient length");
    }
    if (meanLength > 0.0) normal_ = meanGradient / meanLength;

    // Stable orthonormal basis with tangentX=+X, tangentY=+Y at normal=+Z.
    const double sign = std::copysign(1.0, normal_.z());
    const double a = -1.0 / (sign + normal_.z());
    const double b = normal_.x() * normal_.y() * a;
    const Vector3 basisX(1.0 + sign * normal_.x() * normal_.x() * a,
                         sign * b, -sign * normal_.x());
    const Vector3 basisY(b, sign + normal_.y() * normal_.y() * a, -normal_.y());

    const Matrix3 covariance = 0.5 * (covarianceGradient + covarianceGradient.transpose());
    const double xx = basisX.dot(covariance * basisX);
    const double xy = basisX.dot(covariance * basisY);
    const double yy = basisY.dot(covariance * basisY);
    // Rotate to the principal tangent directions so a world-space covariance
    // with off-diagonal terms becomes PBRT's diagonal slope covariance.
    const double angle = 0.5 * std::atan2(2.0 * xy, xx - yy);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    tangentX_ = c * basisX + s * basisY;
    tangentY_ = -s * basisX + c * basisY;
    const double varianceX = tangentX_.dot(covariance * tangentX_);
    const double varianceY = tangentY_.dot(covariance * tangentY_);
    if (!(varianceX > 0.0) || !(varianceY > 0.0)) {
        throw std::invalid_argument("local Beckmann tangent covariance must be positive");
    }
    // The local plane's mean normal derivative is |grad m|. Tangent gradient
    // fluctuations become heightfield slopes after division by that value.
    const double normalDerivative = meanLength > 0.0 ? meanLength : 1.0;
    alphaX_ = std::sqrt(2.0 * varianceX) / normalDerivative;
    alphaY_ = std::sqrt(2.0 * varianceY) / normalDerivative;
    validateAlpha(alphaX_, alphaY_);
}

Vector3 LocalBeckmannVisibleSampler::toLocal(const Vector3& vector) const {
    return Vector3(tangentX_.dot(vector), tangentY_.dot(vector), normal_.dot(vector));
}

Vector3 LocalBeckmannVisibleSampler::toWorld(const Vector3& vector) const {
    return vector.x() * tangentX_ + vector.y() * tangentY_ + vector.z() * normal_;
}

Vector3 LocalBeckmannVisibleSampler::sample(const Vector3& travelDirection, Random& rng) const {
    const Vector3 local = samplePbrtBeckmannVisible(toLocal(travelDirection),
                                                    alphaX_, alphaY_, rng);
    return normalizedOrThrow(toWorld(local));
}

double LocalBeckmannVisibleSampler::pdf(const Vector3& normal,
                                        const Vector3& travelDirection) const {
    return pbrtBeckmannVisiblePdf(toLocal(normal), toLocal(travelDirection),
                                  alphaX_, alphaY_);
}

} // namespace mf

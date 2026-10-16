#pragma once

#include "macrofacet/gpss/FunctionSpaceRaySampler.h"
#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/learned/FirstHitSEModel.h"
#include "macrofacet/transport/RenewalMedium.h"

namespace mf {

struct FirstHitSEFlight {
    Ray ray; // origin at activeDomain entry
    double entryDistance = 0, ell = 0, sigma = 0;
    RayMeanProfile mean;
};

struct FirstHitSESurfaceSample {
    bool enteredDomain = false, hit = false;
    bool failed = false; // GPIS rejection is not an escape to the environment.
    int initialSign = 1;
    double entryDistance = 0, maximumDistance = 0, distance = 0, q = 0, speed = 0;
    Point3 entry = Point3::Zero(), position = Point3::Zero();
    Vector3 gradient = Vector3::Zero(), normal = Vector3::Zero(); // original field sign
    double fieldValue = 0.0;
    Point3 observationPosition = Point3::Zero();
};

// A null model selects the GPIS reference. Both methods share entry geometry,
// cubic profiles and full-gradient reconstruction. No surface-born flights.
class FirstHitSEMedium {
public:
    FirstHitSEMedium(const GPSSField& field, std::shared_ptr<const FirstHitSEModel> model,
        double profileMaximumStep = 0.25, FunctionSpaceSamplerConfig reference = {});
    std::optional<FirstHitSEFlight> beginFlight(const Ray& cameraRay) const;
    FirstHitSESurfaceSample sample(const FirstHitSEFlight& flight, Random& eventRng,
        Random& gradientRng, double physicalTolerance = 1e-10) const;
    FirstHitSESurfaceSample reconstruct(const FirstHitSEFlight& flight,
        const FirstHitSESample& draw, Random& gradientRng) const;
private:
    const GPSSField& field_;
    std::shared_ptr<const FirstHitSEModel> model_;
    double profileMaximumStep_;
    FunctionSpaceRaySampler reference_;
};

// Express the same surface from the path's side: mean and full gradient change
// sign together; kernel, units, domain and interpolation cells do not change.
GPSSField firstHitSideField(const GPSSField& field, int side);

// Versioned, sample-indexed streams isolate camera, event, gradient and tail RNG.
std::uint64_t firstHitStreamSeed(std::uint64_t seed, std::uint64_t pixel,
    std::uint64_t sample, std::uint64_t stream);

} // namespace mf

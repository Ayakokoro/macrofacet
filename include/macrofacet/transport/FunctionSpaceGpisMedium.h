#pragma once
#include "macrofacet/gpss/TungstenFunctionSpaceSampler.h"

namespace mf {
enum class GpisFlightStatus { Escape, Hit, RejectedUpcrossing, InvalidGradient };
struct GpisFlightSample {
    GpisFlightStatus status = GpisFlightStatus::Escape;
    bool enteredDomain = false;
    double entryDistance = 0, maximumDistance = 0, distance = 0;
    Point3 entry = Point3::Zero(), position = Point3::Zero();
    TungstenSurfaceState surface;
    std::uint64_t batches = 0, points = 0;
    int initialSign = 1;
};

class FunctionSpaceGpisMedium {
public:
    FunctionSpaceGpisMedium(const GPSSField& field, TungstenFunctionSpaceConfig config)
        : field_(field), sampler_(field,config) {}
    GpisFlightSample sample(const Ray& ray, std::optional<TungstenSurfaceState> birth,
        TungstenRandom& rng) const;
private:
    const GPSSField& field_;
    TungstenFunctionSpaceSampler sampler_;
};
} // namespace mf

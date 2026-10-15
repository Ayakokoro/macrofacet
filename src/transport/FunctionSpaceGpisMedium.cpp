#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include <stdexcept>

namespace mf {
GpisFlightSample FunctionSpaceGpisMedium::sample(const Ray& input,
    std::optional<TungstenSurfaceState> birth, TungstenRandom& rng) const {
    const Ray ray{input.origin.cast<float>().cast<double>(),
        normalizedOrThrow(input.direction).cast<float>().cast<double>()};
    const auto interval = field_.activeDomain.intersect(ray);
    GpisFlightSample result;
    if (!interval.hit || !(interval.exit>interval.entry)) return result;
    float nearT=static_cast<float>(interval.entry), farT=static_cast<float>(interval.exit);
    result.enteredDomain=true;
    result.entryDistance=nearT; result.maximumDistance=double(farT)-nearT;
    result.entry=ray.origin+double(nearT)*ray.direction.normalized();
    if (!(farT>nearT)) return result;
    bool firstIntersection=true;
    for (;;) {
        const auto batch=sampler_.sampleBatch(ray,nearT,farT,birth,firstIntersection,rng);
        firstIntersection=false;
        ++result.batches; result.points+=sampler_.config().samplePoints;
        if (result.batches==1) result.initialSign=batch.initialSign;
        result.surface=batch.end;
        result.distance=batch.distance-result.entryDistance;
        const float scatterT=std::min(static_cast<float>(batch.distance),farT);
        result.position=(ray.origin.cast<float>()+scatterT*ray.direction.cast<float>()).cast<double>();
        if (!batch.end.gradient.allFinite()) {
            result.status=GpisFlightStatus::InvalidGradient;
            return result;
        }
        if (batch.hit) {
            if (batch.end.gradient.dot(ray.direction)>0) result.status=GpisFlightStatus::RejectedUpcrossing;
            else if (batch.end.gradient.squaredNorm()<static_cast<double>(0.0000001f))
                result.status=GpisFlightStatus::InvalidGradient;
            else result.status=GpisFlightStatus::Hit;
            return result;
        }
        if (batch.distance>=farT) return result;
        const float next=static_cast<float>(batch.distance);
        if (!(next>nearT)) throw std::runtime_error("Tungsten GP step does not advance a float ray");
        nearT=next;
        birth=batch.end;
    }
}
} // namespace mf

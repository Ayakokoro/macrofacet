#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/mathutility/CompensatedSum.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

FlightSample sampleConditionalDeltaTracking(const ConditionalFlightKernel& kernel,
    Random& rng, DdaTrackingDiagnostics* diagnostics, double maximumAge) {
    const double end = maximumAge < 0.0 ? kernel.maximumAgeInDomain() :
        std::min(maximumAge, kernel.maximumAgeInDomain());
    if (!(end >= kernel.currentAge()))
        throw NumericError(NumericStatus::InvalidInput, "invalid conditional flight limit");
    if (end == kernel.currentAge()) return {false, end};
    const ConditionalMajorants bounds=kernel.twoSegmentMajorants(end);
    if (diagnostics) diagnostics->boundIntervals+=bounds.boundIntervals;
    const auto sampleInterval=[&](double begin, double finish, double bound, bool near) {
        CompensatedSum travelled(std::max(kernel.currentAge(),begin));
        if (!(travelled.value()<finish)) return FlightSample{false,finish};
        while (bound > 0.0) {
            const double distance = -std::log1p(-rng.openUniform01()) / bound;
            const double previous=travelled.value();
            if (!(distance < finish - previous)) break;
            if (!(distance>0.0))
                throw NumericError(NumericStatus::NeedHigherPrecision,"conditional candidate distance underflow");
            travelled.add(distance);
            const double age=travelled.value();
            if (!(age<finish)) break;
            if (diagnostics) {
                ++diagnostics->candidates;
                if (age==previous) ++diagnostics->roundedCandidateSteps;
                if (near) ++diagnostics->nearCandidates;
                else ++diagnostics->farCandidates;
            }
            const double hazard = kernel.evaluate(age).hazard.value;
            if (!(hazard <= bound))
                throw NumericError(NumericStatus::InvalidMajorant,
                    "conditional hazard exceeded two-segment majorant");
            if (rng.openUniform01() < hazard / bound) return FlightSample{true, age};
            if (diagnostics) ++diagnostics->nullCollisions;
        }
        return FlightSample{false,finish};
    };
    const auto near=sampleInterval(0.0,bounds.split,bounds.nearMaximum,true);
    if (near.collided) return near;
    const double farBegin=std::max(kernel.currentAge(),bounds.split);
    if (!(farBegin<end)) return {false,end};
    if (bounds.farMaximum*(end-farBegin)<=8192.0)
        return sampleInterval(farBegin,end,bounds.farMaximum,false);

    // Exact Poisson thinning: replace the far rate M by a certified interval
    // rate B<=M and omit the independent rejected process of rate M-B. This
    // changes only null events, never the extinction or the birth observation.
    if (diagnostics) ++diagnostics->adaptiveMajorantFlights;
    struct PendingInterval { double begin,end,maximum; bool tighten; };
    std::vector<PendingInterval> pending;
    for (const auto& root : bounds.farIntervals) {
        if (!(root.end>farBegin)) continue;
        pending.push_back({std::max(root.begin,farBegin),root.end,root.maximum,false});
        while (!pending.empty()) {
            auto interval=pending.back();
            pending.pop_back();
            if (interval.tighten) {
                interval.maximum=std::min(interval.maximum,
                    kernel.intervalMajorant(interval.begin,interval.end));
                if (diagnostics) ++diagnostics->boundIntervals;
            }
            // Refine lazily: intervals after a real collision are never visited.
            while (interval.maximum*(interval.end-interval.begin)>64.0) {
                const double middle=0.5*interval.begin+0.5*interval.end;
                if (!(middle>interval.begin && middle<interval.end)) break;
                pending.push_back({middle,interval.end,interval.maximum,true});
                interval.end=middle;
                interval.maximum=std::min(interval.maximum,
                    kernel.intervalMajorant(interval.begin,interval.end));
                if (diagnostics) ++diagnostics->boundIntervals;
            }
            const auto sample=sampleInterval(interval.begin,interval.end,interval.maximum,false);
            if (sample.collided) return sample;
        }
    }
    return {false, end};
}

} // namespace mf

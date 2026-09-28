#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/mathutility/CompensatedSum.h"
#include "macrofacet/mathutility/NumericPolicy.h"
#include "macrofacet/transport/FlightKernel.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <optional>

namespace mf {

struct ExtinctionSegment {
    double beginAge = 0.0;
    double endAge = 0.0;
    double majorant = 0.0;
    bool nearBirth = false;
};

struct FlightSample {
    bool collided = false;
    double age = 0.0;
};

struct DdaTrackingDiagnostics {
    std::uint64_t candidates = 0;
    std::uint64_t nullCollisions = 0;
    std::uint64_t boundIntervals = 0;
    std::uint64_t nearCandidates = 0;
    std::uint64_t farCandidates = 0;
    std::uint64_t roundedCandidateSteps = 0;
    std::uint64_t adaptiveMajorantFlights = 0;
};

// The cursor supplies certified, ordered intervals only as they are consumed.
// The flight's birth observation and absolute age remain unchanged throughout.
template<class Cursor>
FlightSample sampleSegmentedDeltaTracking(const FlightKernel& kernel, Cursor& cursor,
                                          Random& rng, double end,
                                          DdaTrackingDiagnostics* diagnostics = nullptr) {
    double covered = kernel.currentAge();
    const double boundaryTolerance = 64.0*std::numeric_limits<double>::epsilon()*
        std::max({1.0,std::abs(covered),std::abs(end)});
    while (const std::optional<ExtinctionSegment> next = cursor.next()) {
        const ExtinctionSegment segment = *next;
        if (!(std::abs(segment.beginAge-covered)<=boundaryTolerance &&
              segment.endAge > segment.beginAge &&
              segment.endAge <= end && segment.majorant >= 0.0 &&
              std::isfinite(segment.majorant)))
            throw NumericError(NumericStatus::InvalidMajorant, "invalid extinction segment");
        covered = segment.endAge;
        if (segment.majorant == 0.0) continue;
        CompensatedSum travelled(segment.beginAge);
        while (true) {
            const double previous = travelled.value();
            const double distance = -std::log1p(-rng.openUniform01()) / segment.majorant;
            if (!(distance < segment.endAge - previous)) break;
            if (!(distance > 0.0))
                throw NumericError(NumericStatus::NeedHigherPrecision,
                                   "candidate distance underflow");
            travelled.add(distance);
            const double age = travelled.value();
            if (!(age < segment.endAge)) break;
            if (diagnostics) {
                ++diagnostics->candidates;
                if (age == previous) ++diagnostics->roundedCandidateSteps;
                if (segment.nearBirth) ++diagnostics->nearCandidates;
                else ++diagnostics->farCandidates;
            }
            const double hazard = kernel.evaluate(age).hazard.value;
            if (!(hazard >= 0.0 && hazard <= segment.majorant))
                throw NumericError(NumericStatus::InvalidMajorant,
                                   "extinction exceeds segment majorant");
            if (rng.openUniform01() < hazard / segment.majorant) return {true, age};
            if (diagnostics) ++diagnostics->nullCollisions;
        }
    }
    if (std::abs(covered-end)>boundaryTolerance)
        throw NumericError(NumericStatus::InvalidMajorant,
                           "extinction segments do not cover the flight");
    return {false, end};
}

} // namespace mf

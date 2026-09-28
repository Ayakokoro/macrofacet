#pragma once

#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/SegmentedDeltaTracking.h"
#include <optional>
#include <vector>

namespace mf {

class ConditionalMajorantCursor {
public:
    ConditionalMajorantCursor(const ConditionalFlightKernel& flight, double maximumAge,
                             DdaTrackingDiagnostics* diagnostics = nullptr);
    std::optional<ExtinctionSegment> next();
private:
    struct Pending { double begin, end, maximum; bool tighten; };
    double bound(double begin, double end);
    const ConditionalFlightKernel& flight_;
    DdaTrackingDiagnostics* diagnostics_;
    ConditionalBirthMajorant birth_;
    double age_, end_, correlationLength_;
    bool nearPending_ = true, adapted_ = false;
    std::vector<Pending> pending_;
};

class ConditionalMedium {
public:
    explicit ConditionalMedium(const GPSSField& field) : field_(field) {}
    FlightState startExternal(const Point3& entry, const Vector3& direction,
                              Random& rng) const {
        return startConditionalExterior(field_, entry, direction, rng);
    }
    ConditionalFlightKernel beginFlight(const FlightState& state) const {
        return ConditionalFlightKernel(field_, state);
    }
    ConditionalMajorantCursor sampleRay(const ConditionalFlightKernel& flight,
        double maximumAge = -1.0, DdaTrackingDiagnostics* diagnostics = nullptr) const {
        return ConditionalMajorantCursor(flight, maximumAge, diagnostics);
    }
    FlightSample sample(const ConditionalFlightKernel& flight, Random& rng,
        DdaTrackingDiagnostics* diagnostics = nullptr, double maximumAge = -1.0) const;
private:
    const GPSSField& field_;
};

} // namespace mf

#pragma once

#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include "macrofacet/transport/SegmentedDeltaTracking.h"

namespace mf {

class ClassicMajorantCursor {
public:
    ClassicMajorantCursor(const ClassicFlightKernel& kernel,
                         const DensityMajorantGrid& grid, double areaMajorant,
                         double maximumAge);
    std::optional<ExtinctionSegment> next();
private:
    DensityMajorantGrid::Cursor cells_;
    double areaMajorant_;
};

FlightSample sampleClassicDdaTracking(const ClassicFlightKernel& kernel,
                                     const DensityMajorantGrid& densityMajorant,
                                     double areaMajorant, Random& rng,
                                     double maximumAge = -1.0,
                                     DdaTrackingDiagnostics* diagnostics = nullptr);

} // namespace mf

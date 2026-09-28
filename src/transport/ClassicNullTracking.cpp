#include "macrofacet/transport/ClassicNullTracking.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

ClassicMajorantCursor::ClassicMajorantCursor(const ClassicFlightKernel& kernel,
    const DensityMajorantGrid& grid, double areaMajorant, double maximumAge)
    : cells_(grid, {kernel.state().birthPosition, kernel.state().direction},
             kernel.currentAge(), maximumAge < 0.0 ? kernel.maximumAgeInDomain() :
                 std::min(maximumAge, kernel.maximumAgeInDomain())),
      areaMajorant_(areaMajorant) {
    if (!(areaMajorant_ >= 0.0) || !std::isfinite(areaMajorant_))
        throw NumericError(NumericStatus::InvalidMajorant, "invalid projected-area bound");
}

std::optional<ExtinctionSegment> ClassicMajorantCursor::next() {
    const auto cell = cells_.next();
    if (!cell) return std::nullopt;
    const double product = cell->densityMaximum * areaMajorant_;
    const double rate = product == 0.0 ? 0.0 :
        std::nextafter(product, std::numeric_limits<double>::infinity());
    if (!(rate >= 0.0) || !std::isfinite(rate))
        throw NumericError(NumericStatus::InvalidMajorant, "invalid macrocell majorant");
    return ExtinctionSegment{cell->begin, cell->end, rate, false};
}

FlightSample sampleClassicDdaTracking(const ClassicFlightKernel& kernel,
    const DensityMajorantGrid& densityMajorant, double areaMajorant, Random& rng,
    double maximumAge, DdaTrackingDiagnostics* diagnostics) {
    const double end = maximumAge < 0.0 ? kernel.maximumAgeInDomain() :
        std::min(maximumAge, kernel.maximumAgeInDomain());
    if (!(end >= kernel.currentAge()))
        throw NumericError(NumericStatus::InvalidInput, "invalid classic flight limit");
    ClassicMajorantCursor cursor(kernel, densityMajorant, areaMajorant, end);
    return sampleSegmentedDeltaTracking(kernel, cursor, rng, end, diagnostics);
}

} // namespace mf

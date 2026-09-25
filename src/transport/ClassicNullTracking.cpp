#include "macrofacet/transport/ClassicNullTracking.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

FlightSample sampleClassicNullTracking(const ClassicFlightKernel& kernel,
                                       double certifiedMajorant, Random& rng,
                                       double maximumAge) {
    if (!(certifiedMajorant >= 0.0) || !std::isfinite(certifiedMajorant)) {
        throw NumericError(NumericStatus::InvalidMajorant, "invalid classic majorant");
    }
    double age = kernel.currentAge();
    const double end = maximumAge < 0.0 ? kernel.maximumAgeInDomain() :
        std::min(maximumAge, kernel.maximumAgeInDomain());
    if (!(end >= age))
        throw NumericError(NumericStatus::InvalidInput, "invalid classic flight limit");
    if (certifiedMajorant == 0.0) {
        return {false, end, std::nullopt, std::nullopt, 1.0};
    }
    while (true) {
        age += -std::log1p(-rng.openUniform01()) / certifiedMajorant;
        if (age >= end) {
            return {false, end, std::nullopt, std::nullopt, std::nullopt};
        }
        const double hazard = kernel.evaluate(age).hazard.value;
        if (hazard > certifiedMajorant) {
            throw NumericError(NumericStatus::InvalidMajorant,
                               "classic hazard exceeds its certified majorant");
        }
        if (rng.openUniform01() < hazard / certifiedMajorant) {
            return {true, age, std::nullopt, std::nullopt, std::nullopt};
        }
    }
}

FlightSample sampleClassicDdaTracking(const ClassicFlightKernel& kernel,
                                     const DensityMajorantGrid& densityMajorant,
                                     double areaMajorant, Random& rng,
                                     double maximumAge) {
    if (!(areaMajorant >= 0.0) || !std::isfinite(areaMajorant))
        throw NumericError(NumericStatus::InvalidMajorant, "invalid projected-area bound");
    const Ray ray{kernel.state().birthPosition, kernel.state().direction};
    const double end = maximumAge < 0.0 ? kernel.maximumAgeInDomain() :
        std::min(maximumAge, kernel.maximumAgeInDomain());
    if (!(end >= kernel.currentAge()))
        throw NumericError(NumericStatus::InvalidInput, "invalid classic flight limit");
    const auto segments = densityMajorant.segments(ray, kernel.currentAge(),
                                                   end);
    for (const DensityMajorantSegment& segment : segments) {
        if (segment.densityMaximum == 0.0 || areaMajorant == 0.0) continue;
        const double majorant = std::nextafter(segment.densityMaximum * areaMajorant,
                                               std::numeric_limits<double>::infinity());
        if (!(majorant > 0.0) || !std::isfinite(majorant))
            throw NumericError(NumericStatus::InvalidMajorant, "invalid macrocell majorant");
        double age = segment.begin;
        while (true) {
            const double distance = -std::log1p(-rng.openUniform01()) / majorant;
            if (!(distance < segment.end - age)) break;
            age += distance;
            const double hazard = kernel.evaluate(age).hazard.value;
            if (hazard > majorant)
                throw NumericError(NumericStatus::InvalidMajorant,
                                   "classic hazard exceeds its macrocell majorant");
            if (rng.openUniform01() < hazard / majorant)
                return {true, age, std::nullopt, std::nullopt, std::nullopt};
        }
    }
    return {false, end, std::nullopt, std::nullopt, std::nullopt};
}

} // namespace mf

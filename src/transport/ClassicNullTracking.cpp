#include "macrofacet/transport/ClassicNullTracking.h"
#include <cmath>

namespace mf {

FlightSample sampleClassicNullTracking(const ClassicFlightKernel& kernel,
                                       double certifiedMajorant, Random& rng) {
    if (!(certifiedMajorant >= 0.0) || !std::isfinite(certifiedMajorant)) {
        throw NumericError(NumericStatus::InvalidMajorant, "invalid classic majorant");
    }
    double age = kernel.currentAge();
    if (certifiedMajorant == 0.0) {
        return {false, kernel.maximumAgeInDomain(), std::nullopt, std::nullopt, 1.0};
    }
    while (true) {
        age += -std::log1p(-rng.openUniform01()) / certifiedMajorant;
        if (age >= kernel.maximumAgeInDomain()) {
            return {false, kernel.maximumAgeInDomain(), std::nullopt, std::nullopt, std::nullopt};
        }
        const double hazard = kernel.evaluate(age).hazard.value;
        const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
                                 std::max(1.0, certifiedMajorant);
        if (hazard > certifiedMajorant + tolerance) {
            throw NumericError(NumericStatus::InvalidMajorant,
                               "classic hazard exceeds its certified majorant");
        }
        if (rng.openUniform01() < hazard / certifiedMajorant) {
            return {true, age, std::nullopt, std::nullopt, std::nullopt};
        }
    }
}

} // namespace mf


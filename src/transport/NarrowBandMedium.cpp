#include "macrofacet/transport/NarrowBandMedium.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include <cmath>
#include <stdexcept>

namespace mf {

NarrowBandMedium::NarrowBandMedium(
    const GPSSField& field, const MaterialConfig& material, ScalarFieldPtr density,
    std::shared_ptr<const DensityMajorantGrid> majorantGrid,
    std::optional<double> preparedAreaMajorant)
    : field_(field), material_(material), density_(std::move(density)),
      majorantGrid_(std::move(majorantGrid)) {
    if (!density_ || !majorantGrid_)
        throw std::invalid_argument("DDA tracking requires density and majorant grids");
    // Prepared renders supply the certified bound from field preparation.
    // Direct library users can still construct a medium without that stage.
    areaMajorant_ = preparedAreaMajorant.value_or(0.0);
    if (preparedAreaMajorant) {
        if (!(areaMajorant_ >= 0.0) || !std::isfinite(areaMajorant_))
            throw std::invalid_argument("invalid prepared projected-area majorant");
    } else {
        areaMajorant_ = classicAreaMajorant(field_, material_, field_.activeDomain,
                                            Vector3::UnitZ());
    }
}

FlightState NarrowBandMedium::startExternal(const Point3& entry,
                                            const Vector3& direction) const {
    return startExternalFlight(entry, direction);
}

ClassicFlightKernel NarrowBandMedium::beginFlight(const FlightState& state) const {
    return ClassicFlightKernel(field_, material_, state, density_);
}

FlightSample NarrowBandMedium::sample(const ClassicFlightKernel& flight, Random& rng,
                                      const NumericPolicy& numeric,
                                      DdaTrackingDiagnostics* diagnostics,
                                      double maximumAge) const {
    (void)numeric;
    return sampleClassicDdaTracking(flight, *majorantGrid_, areaMajorant_, rng,
                                    maximumAge, diagnostics);
}

} // namespace mf

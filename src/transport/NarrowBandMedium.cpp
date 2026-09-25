#include "macrofacet/transport/NarrowBandMedium.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {

NarrowBandMedium::NarrowBandMedium(
    const GPSSField& field, const MaterialConfig& material, ScalarFieldPtr density,
    bool surfaceBand, std::shared_ptr<const DensityMajorantGrid> majorantGrid)
    : field_(field), material_(material), density_(std::move(density)),
      surfaceBand_(surfaceBand), majorantGrid_(std::move(majorantGrid)) {
    if (surfaceBand_ && density_ && !majorantGrid_) {
        const ScalarBounds bounds = density_->bounds(field_.activeDomain);
        if (!bounds.certified || !(bounds.maximumValue >= 0.0) ||
            !std::isfinite(bounds.maximumValue)) {
            throw NumericError(NumericStatus::InvalidMajorant,
                               "density field has no finite certified upper bound");
        }
        fallbackDensityMaximum_ = bounds.maximumValue;
    }
}

FlightState NarrowBandMedium::startExternal(ModelMode mode, ExternalPolicy policy,
                                            const Point3& entry, const Vector3& direction,
                                            Random& rng, const NumericPolicy& numeric) const {
    if (mode == ModelMode::Conditional29 && policy == ExternalPolicy::SampledExterior) {
        if (surfaceBand_ && density_ && !(density_->sample(entry) > 0.0)) {
            throw std::invalid_argument(
                "sampled_exterior starts outside the transport band; use original_macrofacet");
        }
        return sampleExteriorFlight(field_, entry, direction, rng, numeric);
    }
    return startExternalFlight(entry, direction);
}

std::unique_ptr<FlightKernel> NarrowBandMedium::beginFlight(
    ModelMode mode, const FlightState& state, ExternalPolicy policy,
    const NumericPolicy& numeric) const {
    return makeFlightKernel(mode, field_, material_, state, policy, numeric, density_);
}

FlightSample NarrowBandMedium::sample(const FlightKernel& flight, Random& rng,
                                      const NumericPolicy& numeric,
                                      TrackingDiagnostics* diagnostics, int initialCells,
                                      double maximumAge) const {
    if (surfaceBand_ && density_ && flight.mode() == ModelMode::Classic) {
        const auto& classic = static_cast<const ClassicFlightKernel&>(flight);
        const double area = classicAreaMajorant(field_, material_, field_.activeDomain,
                                                 flight.state().direction, numeric);
        // The macrocell grid certified all density bounds once at construction.
        // Re-querying the whole density domain here would scan the NanoVDB on
        // every flight even though the DDA tracker only uses those cached cells.
        if (majorantGrid_)
            return sampleClassicDdaTracking(classic, *majorantGrid_, area, rng, maximumAge);

        const double majorant = std::nextafter(fallbackDensityMaximum_ * area,
                                               std::numeric_limits<double>::infinity());
        return sampleClassicNullTracking(classic, majorant, rng, maximumAge);
    }
    return sampleFlight(flight, rng, numeric, diagnostics, initialCells, maximumAge);
}

} // namespace mf

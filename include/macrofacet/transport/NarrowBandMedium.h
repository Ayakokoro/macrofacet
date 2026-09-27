#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/macrofacet/MaterialConfig.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include <optional>

namespace mf {

// Classic transport through a baked density field using DDA null tracking.
class NarrowBandMedium {
public:
    NarrowBandMedium(const GPSSField& field, const MaterialConfig& material,
                     ScalarFieldPtr density,
                     std::shared_ptr<const DensityMajorantGrid> majorantGrid,
                     std::optional<double> preparedAreaMajorant = std::nullopt);

    FlightState startExternal(const Point3& entry, const Vector3& direction) const;
    ClassicFlightKernel beginFlight(const FlightState& state) const;
    FlightSample sample(const ClassicFlightKernel& flight, Random& rng,
                        const NumericPolicy& numeric,
                        DdaTrackingDiagnostics* diagnostics = nullptr,
                        double maximumAge = -1.0) const;

private:
    const GPSSField& field_;
    const MaterialConfig& material_;
    ScalarFieldPtr density_;
    std::shared_ptr<const DensityMajorantGrid> majorantGrid_;
    double areaMajorant_ = 0.0;
};

} // namespace mf

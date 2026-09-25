#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/macrofacet/MaterialConfig.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

namespace mf {

// A transport model whose baked density grid defines the active band. The GP
// prior is used only for in-band projected areas and conditional statistics.
// A null density field retains the analytic fallback used by core-only tests.
class NarrowBandMedium {
public:
    NarrowBandMedium(const GPSSField& field, const MaterialConfig& material,
                     ScalarFieldPtr density = nullptr, bool surfaceBand = false,
                     std::shared_ptr<const DensityMajorantGrid> majorantGrid = nullptr);

    FlightState startExternal(ModelMode mode, ExternalPolicy policy,
                              const Point3& entry, const Vector3& direction,
                              Random& rng, const NumericPolicy& numeric) const;
    std::unique_ptr<FlightKernel> beginFlight(ModelMode mode, const FlightState& state,
                                               ExternalPolicy policy,
                                               const NumericPolicy& numeric) const;
    FlightSample sample(const FlightKernel& flight, Random& rng,
                        const NumericPolicy& numeric, TrackingDiagnostics* diagnostics,
                        int initialCells, double maximumAge = -1.0) const;

private:
    const GPSSField& field_;
    const MaterialConfig& material_;
    ScalarFieldPtr density_;
    bool surfaceBand_ = false;
    std::shared_ptr<const DensityMajorantGrid> majorantGrid_;
    double fallbackDensityMaximum_ = 0.0;
};

} // namespace mf

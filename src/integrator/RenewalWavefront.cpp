#include "macrofacet/integrator/RenewalWavefront.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {
namespace {
Spectrum emission(const Vector3& direction, const std::string& environment) {
    if (environment == "unit_white") return Spectrum::Ones();
    const double up = 0.5*(normalizedOrThrow(direction).z()+1.0);
    return Spectrum(0.15+0.85*up,0.2+0.7*up,0.35+0.55*up);
}
struct Slot {
    std::size_t pixel = 0;
    int sample = 0, depth = 0;
    bool occupied = false, newSample = true;
    Random rng;
    Ray ray;
    Spectrum throughput = Spectrum::Ones();
    std::optional<Vector3> gradient;
    std::optional<RenewalFlight> flight;
    std::size_t segment = 0;
    double remaining = 0;
};
}

RenderedImage renderRenewalWavefront(const ExperimentConfig& config,
    std::atomic<std::uint64_t>* completedCameraRays) {
    RenderedImage image;
    image.width = config.render.width; image.height = config.render.height;
    const auto pixelCount = static_cast<std::size_t>(image.width)*image.height;
    image.pixels.assign(pixelCount,Spectrum::Zero());
    const int capacity = static_cast<int>(std::min<std::size_t>(config.renewal.batchSize,pixelCount));
    RenewalBatchSession network(*config.renewal.model,capacity,config.renewal.resolvedBackend);
    const RenewalMedium medium(config.field,*config.renewal.model,config.renewal.profileMaximumStep,
                               config.renewal.profileMode);
    auto& statistics = image.statistics;
    statistics.renewal.backend = network.backend();
    const Vector3 forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
    Vector3 right, up;
    orthonormalComplement(forward,right,up);
    const double aspect = static_cast<double>(image.width)/image.height;
    const double scale = std::tan(config.render.verticalFovDegrees*kPi/360.0);
    std::vector<Slot> slots(static_cast<std::size_t>(capacity));
    std::size_t nextPixel = 0;
    const auto finish = [&](Slot& slot, const Spectrum& radiance) {
        // Each pixel retains its original RNG stream and accumulation order,
        // irrespective of slot reuse, batch size or another ray's path length.
        image.pixels[slot.pixel] += radiance;
        statistics.accumulatedPathDepth += slot.depth;
        slot.flight.reset(); slot.newSample = true;
        if (++slot.sample == config.render.samplesPerPixel) {
            image.pixels[slot.pixel] /= config.render.samplesPerPixel;
            slot.occupied = false;
        }
        if (completedCameraRays) completedCameraRays->fetch_add(1,std::memory_order_relaxed);
    };
    const auto escape = [&](Slot& slot) {
        ++statistics.escapedPaths;
        finish(slot,slot.throughput.cwiseProduct(emission(slot.ray.direction,config.render.environment)));
    };
    std::vector<int> initializeIds, activeIds, hitIds;
    std::vector<RayMeanSegment> firstSegments, activeSegments;
    std::vector<RayStartCondition> starts;
    std::vector<double> hitCoordinates;
    for (;;) {
        initializeIds.clear(); firstSegments.clear(); starts.clear();
        activeIds.clear(); activeSegments.clear();
        for (int id = 0; id < capacity; ++id) {
            auto& slot = slots[id];
            // Refill slots immediately, including camera misses, without making
            // a device request for rays that never enter the active domain.
            while (!slot.flight) {
                if (!slot.occupied) {
                    if (nextPixel == pixelCount) break;
                    slot.pixel = nextPixel++; slot.sample = 0;
                    slot.rng = Random(config.seed+0x9e3779b97f4a7c15ULL*(slot.pixel+1));
                    slot.occupied = true; slot.newSample = true;
                }
                if (slot.newSample) {
                    slot.newSample = false; slot.depth = 0;
                    slot.gradient.reset(); slot.throughput = Spectrum::Ones();
                    ++statistics.paths;
                    const int px = static_cast<int>(slot.pixel%image.width);
                    const int py = static_cast<int>(slot.pixel/image.width);
                    const double u = (2.0*((px+slot.rng.openUniform01())/image.width)-1.0)*aspect*scale;
                    const double v = (1.0-2.0*((py+slot.rng.openUniform01())/image.height))*scale;
                    const Vector3 direction = normalizedOrThrow(forward+u*right+v*up);
                    slot.ray = {config.render.cameraPosition,normalizedOrThrow(direction)};
                    const auto domain = config.field.activeDomain.intersect(slot.ray);
                    if (!domain.hit || !(domain.exit > domain.entry)) { escape(slot); continue; }
                    slot.ray.origin += domain.entry*slot.ray.direction;
                    slot.ray.origin = slot.ray.origin.cwiseMax(config.field.activeDomain.minimum)
                        .cwiseMin(config.field.activeDomain.maximum);
                }
                slot.flight = medium.beginFlight(slot.ray,slot.gradient);
                if (!slot.flight) { escape(slot); continue; }
                slot.segment = 0; slot.remaining = -std::log(slot.rng.openUniform01());
                ++statistics.renewal.flights;
                initializeIds.push_back(id);
                firstSegments.push_back(slot.flight->mean.segment(0));
                starts.push_back(slot.flight->start);
            }
            if (slot.flight) {
                activeIds.push_back(id);
                activeSegments.push_back(slot.flight->mean.segment(slot.segment));
            }
        }
        if (activeIds.empty()) break;
        if (!initializeIds.empty()) {
            network.initialize(initializeIds,firstSegments,starts);
            ++statistics.renewal.initializationBatches;
        }
        const auto rates = network.evaluate(activeIds,activeSegments);
        ++statistics.renewal.segmentBatches;
        statistics.renewal.segments += activeIds.size();
        statistics.renewal.maximumBatchSize = std::max(statistics.renewal.maximumBatchSize,
            static_cast<std::uint64_t>(activeIds.size()));
        hitIds.clear(); hitCoordinates.clear();
        for (std::size_t i = 0; i < activeIds.size(); ++i) {
            auto& slot = slots[activeIds[i]];
            const auto& segment = activeSegments[i];
            const RenewalHazardSegment hazard(segment.begin,segment.end,rates[i]);
            const double mass = hazard.integral(0,1);
            if (mass > 0 && slot.remaining <= mass) {
                hitIds.push_back(activeIds[i]);
                hitCoordinates.push_back(hazard.inverse(slot.remaining,0,1,
                    config.numeric.distanceAbsoluteTolerance/slot.flight->ell));
            } else {
                slot.remaining -= mass;
                if (!slot.flight->mean.hasSegment(++slot.segment)) escape(slot);
            }
        }
        if (!hitIds.empty()) {
            const auto mixtures = network.mixture(hitIds,hitCoordinates);
            ++statistics.renewal.mixtureBatches;
            statistics.renewal.mixtureQueries += hitIds.size();
            for (std::size_t i = 0; i < hitIds.size(); ++i) {
                auto& slot = slots[hitIds[i]];
                const auto hit = medium.sampleHit(*slot.flight,slot.segment,hitCoordinates[i],mixtures[i],slot.rng);
                ++statistics.realCollisions; ++slot.depth;
                slot.throughput = slot.throughput.cwiseProduct(conductorFresnel(
                    -slot.ray.direction.dot(hit.normal),config.material.conductor));
                const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(slot.ray.direction,hit.normal));
                if (!(outgoing.dot(hit.gradient) > 0))
                    throw std::runtime_error("Renewal reflection must leave the sampled surface");
                slot.ray = {hit.position,outgoing}; slot.gradient = hit.gradient;
                slot.flight.reset();
                if (slot.depth >= config.render.rouletteStartDepth) {
                    const double probability = std::clamp(slot.throughput.maxCoeff(),0.05,0.95);
                    if (slot.rng.openUniform01() >= probability) {
                        ++statistics.rouletteTerminations; finish(slot,Spectrum::Zero()); continue;
                    }
                    slot.throughput /= probability;
                }
                if (slot.depth == config.render.safetyDepthCap) {
                    ++statistics.safetyCapTerminations; finish(slot,Spectrum::Zero());
                }
            }
        }
    }
    return image;
}
} // namespace mf

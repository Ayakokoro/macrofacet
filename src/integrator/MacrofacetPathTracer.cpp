#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/integrator/RenewalWavefront.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <thread>

namespace mf {
namespace {

Spectrum environmentEmission(const Vector3& direction, const std::string& environment) {
    if (environment == "unit_white") return Spectrum::Ones();
    const double up = 0.5 * (normalizedOrThrow(direction).z() + 1.0);
    return Spectrum(0.15 + 0.85 * up, 0.2 + 0.7 * up, 0.35 + 0.55 * up);
}

int resolveWorkerCount(int requested, int rows) {
    if (requested > 0) return std::max(1, std::min(requested, rows));
    const unsigned hardware = std::thread::hardware_concurrency();
    const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
    return std::max(1, std::min(available, rows));
}

Spectrum traceRenewalFlights(Ray ray, std::optional<Vector3> gradient,
    Spectrum throughput, int startDepth, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    for (int depth = startDepth; depth < config.render.safetyDepthCap; ++depth) {
        const auto flight = medium.beginFlight(ray, gradient);
        const auto hit = flight ? medium.sample(*flight, rng,
            config.numeric.distanceAbsoluteTolerance, &statistics.renewal) : RenewalSurfaceSample{};
        if (!hit.hit) {
            ++statistics.escapedPaths;
            statistics.accumulatedPathDepth += depth;
            return throughput.cwiseProduct(environmentEmission(ray.direction, config.render.environment));
        }
        ++statistics.realCollisions;
        throughput = throughput.cwiseProduct(conductorFresnel(-ray.direction.dot(hit.normal), config.material.conductor));
        const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(ray.direction, hit.normal));
        if (!(outgoing.dot(hit.gradient) > 0))
            throw std::runtime_error("Renewal reflection must leave the sampled surface");
        // The sampled geometry is part of the target model: no inverse hit PDF
        // or extra Rice factor. A new direction starts B with the FULL gradient.
        ray = {hit.position, outgoing};
        gradient = hit.gradient;
        if (depth+1 >= config.render.rouletteStartDepth) {
            const double probability = std::clamp(throughput.maxCoeff(), 0.05, 0.95);
            if (rng.openUniform01() >= probability) {
                ++statistics.rouletteTerminations;
                statistics.accumulatedPathDepth += depth+1;
                return Spectrum::Zero();
            }
            throughput /= probability;
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth += config.render.safetyDepthCap;
    return Spectrum::Zero();
}

Spectrum traceRenewalCameraPath(const Ray& input, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    Ray ray{input.origin, normalizedOrThrow(input.direction)};
    const auto interval = config.field.activeDomain.intersect(ray);
    if (!interval.hit || !(interval.exit > interval.entry)) {
        ++statistics.escapedPaths;
        return environmentEmission(ray.direction, config.render.environment);
    }
    ray.origin += interval.entry*ray.direction;
    ray.origin = ray.origin.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
    return traceRenewalFlights(ray,std::nullopt,Spectrum::Ones(),0,config,medium,rng,statistics);
}

// Continue an existing path with the same classic/conditional kernel used by
// camera paths. startDepth counts reflections that have already been applied.
Spectrum traceClassicFlights(FlightState state, Spectrum throughput, int startDepth,
    const ExperimentConfig& config, const NarrowBandMedium& medium, Random& rng,
    RenderStatistics& statistics) {
    const bool conditional = config.transportMode == "global_conditional";
    const ConditionalMedium conditionalMedium(config.field);
    int depth = startDepth;
    for (; depth < config.render.safetyDepthCap; ++depth) {
        try {
            FlightSample flight;
            Vector3 collisionGradient = Vector3::Zero();
            if (conditional) {
                const ConditionalFlightKernel kernel = conditionalMedium.beginFlight(state);
                flight = conditionalMedium.sample(kernel, rng, &statistics.tracking);
                if (flight.collided)
                    collisionGradient = sampleCollisionGradient(kernel, flight.age,
                                                                 rng, config.numeric);
            } else {
                ++statistics.classicFlights;
                const ClassicFlightKernel kernel = medium.beginFlight(state);
                flight = medium.sample(kernel, rng, config.numeric, &statistics.tracking);
            }
            if (!flight.collided) {
                ++statistics.escapedPaths;
                statistics.accumulatedPathDepth += depth;
                return throughput.cwiseProduct(
                    environmentEmission(state.direction, config.render.environment));
            }
            const Point3 hit = state.birthPosition + flight.age * state.direction;
            ++statistics.realCollisions;
            if (conditional) {
                const Vector3 normal = normalizedOrThrow(collisionGradient);
                throughput = throughput.cwiseProduct(conductorFresnel(
                    -state.direction.dot(normal), config.material.conductor));
                const Vector3 outgoing = reflectTravelDirection(state.direction, normal);
                state = startClassicCollisionFlight(hit, outgoing);
                state.birthValue = 0.0;
                state.birthGradient = collisionGradient;
            } else {
                const double mixture = config.classicPhaseProposal == "paper_mixture"
                    ? config.beckmannMixtureWeight : 0.0;
                const bool useTargetVndf = config.classicPhaseProposal == "target_vndf";
                ConductorPhase phase(config.field, config.material, hit, mixture, useTargetVndf);
                const PhaseSample sample = phase.samplePhase(state.direction, rng);
                throughput = throughput.cwiseProduct(sample.throughputWeight);
                state = startClassicCollisionFlight(hit, sample.direction);
            }
            if (depth + 1 >= config.render.rouletteStartDepth) {
                const double continuation = std::clamp(throughput.maxCoeff(), 0.05, 0.95);
                if (rng.openUniform01() >= continuation) {
                    ++statistics.rouletteTerminations;
                    statistics.accumulatedPathDepth += depth + 1;
                    return Spectrum::Zero();
                }
                throughput /= continuation;
            }
        } catch (const NumericError&) {
            if (conditional) throw; // A failed majorant must not become a black sample.
            ++statistics.numericalFailures;
            statistics.accumulatedPathDepth += depth;
            return Spectrum::Zero();
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth += depth;
    return Spectrum::Zero();
}

class HybridFirstHitTransport {
    const ExperimentConfig& config_;
    GPSSField negativeField_;
    FirstHitSEMedium first_;
    std::unique_ptr<RenewalMedium> positive_, negative_;
    std::unique_ptr<NarrowBandMedium> local_;
public:
    explicit HybridFirstHitTransport(const ExperimentConfig& config)
        : config_(config), negativeField_(firstHitSideField(config.field,-1)),
          first_(config.field,config.firstHit.model,config.firstHit.profileMaximumStep,config.firstHit.reference) {
        if (config.continuationMethod == "classic_local") {
            // This is the scene's existing local transport model. Its baked
            // density and tangent NDF keep the original field orientation;
            // it does not carry the first segment's GP conditioning forward.
            local_ = std::make_unique<NarrowBandMedium>(config.field,config.material,
                config.mediumDensity,config.densityMajorantGrid,config.preparedAreaMajorant);
        } else {
            positive_ = std::make_unique<RenewalMedium>(config.field,*config.renewal.model,
                config.renewal.profileMaximumStep,config.renewal.profileMode);
            negative_ = std::make_unique<RenewalMedium>(negativeField_,*config.renewal.model,
                config.renewal.profileMaximumStep,config.renewal.profileMode);
        }
    }

    Spectrum trace(const Ray& input, Random& eventRng, Random& gradientRng, Random& continuationRng,
        RenderStatistics& statistics, FirstHitRenderRecord* record = nullptr) const {
        const Ray ray{input.origin,normalizedOrThrow(input.direction)};
        const auto flight = first_.beginFlight(ray);
        FirstHitSESurfaceSample hit;
        if (flight) {
            ++statistics.firstHitFlights;
            hit = first_.sample(*flight,eventRng,gradientRng,config_.numeric.distanceAbsoluteTolerance);
        } else ++statistics.firstHitDomainMisses;
        if (record) { record->cameraRay = ray; record->first = hit; }
        if (!hit.hit) {
            if (flight) ++statistics.firstHitEscapes;
            ++statistics.escapedPaths;
            return environmentEmission(ray.direction,config_.render.environment);
        }
        ++statistics.firstHits; ++statistics.realCollisions;
        if (hit.initialSign < 0) ++statistics.firstHitUpcrossings;
        RenewalPathState state;
        state.throughput = conductorFresnel(-ray.direction.dot(hit.normal),config_.material.conductor);
        state.ray = {hit.position,normalizedOrThrow(reflectTravelDirection(ray.direction,hit.normal))};
        state.gradient = hit.initialSign*hit.gradient;
        // The first event already sampled survival, distance and the collision
        // gradient. Only Fresnel is applied here, with no second phase sample.
        if (state.depth >= config_.render.rouletteStartDepth) {
            const double p = std::clamp(state.throughput.maxCoeff(),0.05,0.95);
            if (continuationRng.openUniform01() >= p) {
                ++statistics.rouletteTerminations; statistics.accumulatedPathDepth += state.depth;
                return Spectrum::Zero();
            }
            state.throughput /= p;
        }
        if (local_) return traceClassicFlights(startClassicCollisionFlight(state.ray.origin,state.ray.direction),
            state.throughput,state.depth,config_,*local_,continuationRng,statistics);
        return traceRenewalFromSurface(state,config_,hit.initialSign > 0 ? *positive_ : *negative_,
            continuationRng,statistics);
    }
};

} // namespace

Spectrum traceRenewalFromSurface(const RenewalPathState& state, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    if (state.depth < 1 || !state.gradient.allFinite() || !state.throughput.allFinite() ||
        !(state.ray.direction.dot(state.gradient) > 0))
        throw std::invalid_argument("invalid outward renewal continuation state");
    Ray ray{state.ray.origin,normalizedOrThrow(state.ray.direction)};
    return traceRenewalFlights(ray,state.gradient,state.throughput,state.depth,config,medium,rng,statistics);
}

Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, Random& rng,
                         RenderStatistics& statistics) {
    if (config.transportMode == "hybrid_first_hit") {
        ExperimentConfig prepared = config;
        prepareRenewalModel(prepared); prepareFirstHitModel(prepared);
        const HybridFirstHitTransport hybrid(prepared);
        const auto seed = rng.integer();
        Random eventRng(firstHitStreamSeed(seed,0,0,1)), gradientRng(firstHitStreamSeed(seed,0,0,2)),
               continuationRng(firstHitStreamSeed(seed,0,0,3));
        return hybrid.trace(initialRay,eventRng,gradientRng,continuationRng,statistics);
    }
    if (config.transportMode == "neural_renewal") {
        ExperimentConfig prepared = config;
        prepareRenewalModel(prepared);
        const RenewalMedium medium(prepared.field, *prepared.renewal.model, prepared.renewal.profileMaximumStep,
                                   prepared.renewal.profileMode);
        return traceRenewalCameraPath(initialRay, prepared, medium, rng, statistics);
    }
    const NarrowBandMedium medium(config.field, config.material, config.mediumDensity,
                                  config.densityMajorantGrid, config.preparedAreaMajorant);
    return traceCameraPath(initialRay, config, medium, rng, statistics);
}

Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, const NarrowBandMedium& medium,
                         Random& rng, RenderStatistics& statistics) {
    if (config.transportMode == "neural_renewal" || config.transportMode == "hybrid_first_hit")
        return traceCameraPath(initialRay, config, rng, statistics);
    const Vector3 initialDirection = normalizedOrThrow(initialRay.direction);
    const DomainInterval firstInterval = config.field.activeDomain.intersect(
        {initialRay.origin, initialDirection});
    if (!firstInterval.hit) {
        ++statistics.escapedPaths;
        return environmentEmission(initialDirection, config.render.environment);
    }
    const Point3 entry = initialRay.origin + firstInterval.entry * initialDirection;
    const bool conditional = config.transportMode == "global_conditional";
    const ConditionalMedium conditionalMedium(config.field);
    FlightState state = conditional
        ? conditionalMedium.startExternal(entry, initialDirection, rng)
        : medium.startExternal(entry, initialDirection);
    return traceClassicFlights(state,Spectrum::Ones(),0,config,medium,rng,statistics);
}

RenderedImage renderAnalyticScene(const ExperimentConfig& config,
                                 std::atomic<std::uint64_t>* completedCameraRays) {
    // Prepare the local extinction bound before tracing any paths. All worker
    // threads share this immutable medium and its baked DDA majorant.
    ExperimentConfig prepared = config;
    std::unique_ptr<NarrowBandMedium> medium;
    std::unique_ptr<RenewalMedium> renewal;
    std::unique_ptr<HybridFirstHitTransport> hybrid;
    if (config.transportMode == "hybrid_first_hit") {
        prepareRenewalModel(prepared); prepareFirstHitModel(prepared);
        hybrid = std::make_unique<HybridFirstHitTransport>(prepared);
    } else if (config.transportMode == "neural_renewal") {
        prepareRenewalModel(prepared);
        if (prepared.renewal.resolvedBackend != "scalar")
            return renderRenewalWavefront(prepared,completedCameraRays);
        renewal = std::make_unique<RenewalMedium>(prepared.field, *prepared.renewal.model,
                                                prepared.renewal.profileMaximumStep, prepared.renewal.profileMode);
    } else {
        medium = std::make_unique<NarrowBandMedium>(config.field, config.material, config.mediumDensity,
                                                   config.densityMajorantGrid, config.preparedAreaMajorant);
    }
    RenderedImage result;
    result.width = config.render.width;
    result.height = config.render.height;
    result.pixels.assign(static_cast<std::size_t>(result.width * result.height), Spectrum::Zero());
    if (hybrid && config.render.writeFirstHitSamples) {
        const auto count = result.pixels.size()*static_cast<std::size_t>(config.render.samplesPerPixel);
        if (count > 1000000) throw std::invalid_argument("first-hit CSV diagnostics are limited to 1,000,000 camera samples");
        result.firstHitRecords.resize(count);
    }
    const Vector3 forward = normalizedOrThrow(config.render.cameraTarget - config.render.cameraPosition);
    Vector3 right, up;
    orthonormalComplement(forward, right, up);
    const double aspect = static_cast<double>(result.width) / result.height;
    const double scale = std::tan(config.render.verticalFovDegrees * kPi / 360.0);
    // Every pixel seeds its own stream from its index alone, so distributing rows over
    // workers leaves the image bit-identical to the serial result.
    const int workers = resolveWorkerCount(config.render.threadCount, result.height);
    std::vector<RenderStatistics> perWorker(static_cast<std::size_t>(workers));
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(workers));
    std::atomic<int> nextRow{0};
    std::atomic<bool> aborted{false};

    const auto renderRow = [&](int worker, int y) {
        RenderStatistics& statistics = perWorker[static_cast<std::size_t>(worker)];
        for (int x = 0; x < result.width; ++x) {
            Spectrum sum = Spectrum::Zero();
            const std::uint64_t pixelIndex = static_cast<std::uint64_t>(y * result.width + x);
            Random rng(config.seed + 0x9e3779b97f4a7c15ULL * (pixelIndex + 1));
            for (int sample = 0; sample < config.render.samplesPerPixel; ++sample) {
                std::optional<Random> cameraRng;
                if (hybrid) cameraRng.emplace(firstHitStreamSeed(config.seed,pixelIndex,sample,0));
                Random& jitterRng = hybrid ? *cameraRng : rng;
                const double px = (2.0 * ((x + jitterRng.openUniform01()) / result.width) - 1.0) *
                                  aspect * scale;
                const double py = (1.0 - 2.0 * ((y + jitterRng.openUniform01()) / result.height)) * scale;
                const Vector3 direction = normalizedOrThrow(forward + px * right + py * up);
                ++statistics.paths;
                const Ray ray{config.render.cameraPosition, direction};
                if (hybrid) {
                    Random eventRng(firstHitStreamSeed(config.seed,pixelIndex,sample,1));
                    Random gradientRng(firstHitStreamSeed(config.seed,pixelIndex,sample,2));
                    Random continuationRng(firstHitStreamSeed(config.seed,pixelIndex,sample,3));
                    auto* record = result.firstHitRecords.empty() ? nullptr :
                        &result.firstHitRecords[pixelIndex*config.render.samplesPerPixel+sample];
                    Spectrum radiance;
                    try {
                        radiance = hybrid->trace(ray,eventRng,gradientRng,continuationRng,statistics,record);
                    } catch (const std::exception& error) {
                        throw std::runtime_error("hybrid_first_hit pixel=("+std::to_string(x)+","+
                            std::to_string(y)+"), sample="+std::to_string(sample)+", seed="+
                            std::to_string(config.seed)+": "+error.what());
                    }
                    if (record) record->radiance = radiance;
                    sum += radiance;
                } else sum += renewal ? traceRenewalCameraPath(ray, prepared, *renewal, rng, statistics)
                                      : traceCameraPath(ray, config, *medium, rng, statistics);
                if (completedCameraRays)
                    completedCameraRays->fetch_add(1, std::memory_order_relaxed);
            }
            result.pixels[static_cast<std::size_t>(y * result.width + x)] =
                sum / config.render.samplesPerPixel;
        }
    };

    // Dynamic row assignment: cost per row varies by orders of magnitude across a
    // image, so static chunks would leave workers idle.
    const auto drainRows = [&](int worker) {
        for (;;) {
            const int y = nextRow.fetch_add(1, std::memory_order_relaxed);
            if (y >= result.height || aborted.load(std::memory_order_relaxed)) return;
            renderRow(worker, y);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers) - 1);
    try {
        for (int worker = 1; worker < workers; ++worker) {
            pool.emplace_back([&, worker] {
                try {
                    drainRows(worker);
                } catch (...) {
                    failures[static_cast<std::size_t>(worker)] = std::current_exception();
                    aborted.store(true, std::memory_order_relaxed);
                }
            });
        }
    } catch (...) {
        aborted.store(true, std::memory_order_relaxed);
        for (std::thread& thread : pool) {
            if (thread.joinable()) thread.join();
        }
        throw;
    }
    try {
        drainRows(0);
    } catch (...) {
        failures[0] = std::current_exception();
        aborted.store(true, std::memory_order_relaxed);
    }
    for (std::thread& thread : pool) thread.join();
    for (const std::exception_ptr& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
    for (const RenderStatistics& statistics : perWorker) mergeInto(result.statistics, statistics);
    return result;
}

void writePfm(const std::filesystem::path& path, const RenderedImage& image) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open PFM output");
    stream << "PF\n" << image.width << ' ' << image.height << "\n-1.0\n";
    for (int y = image.height - 1; y >= 0; --y) {
        for (int x = 0; x < image.width; ++x) {
            const Spectrum& pixel = image.pixels[static_cast<std::size_t>(y * image.width + x)];
            const float rgb[3]{static_cast<float>(pixel.x()), static_cast<float>(pixel.y()),
                               static_cast<float>(pixel.z())};
            stream.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
    }
}

void writeBmpPreview(const std::filesystem::path& path, const RenderedImage& image) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open BMP preview output");
    auto write16 = [&stream](std::uint16_t value) {
        const unsigned char bytes[2]{static_cast<unsigned char>(value),
                                     static_cast<unsigned char>(value >> 8)};
        stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    };
    auto write32 = [&stream](std::uint32_t value) {
        const unsigned char bytes[4]{static_cast<unsigned char>(value),
                                     static_cast<unsigned char>(value >> 8),
                                     static_cast<unsigned char>(value >> 16),
                                     static_cast<unsigned char>(value >> 24)};
        stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    };
    const std::uint32_t rowBytes = static_cast<std::uint32_t>((image.width * 3 + 3) & ~3);
    const std::uint32_t pixelBytes = rowBytes * static_cast<std::uint32_t>(image.height);
    stream.write("BM", 2);
    write32(54 + pixelBytes);
    write32(0);
    write32(54);
    write32(40);
    write32(static_cast<std::uint32_t>(image.width));
    write32(static_cast<std::uint32_t>(image.height));
    write16(1);
    write16(24);
    write32(0);
    write32(pixelBytes);
    write32(2835);
    write32(2835);
    write32(0);
    write32(0);
    auto encode = [](double linear) {
        const double x = std::clamp(linear, 0.0, 1.0);
        const double srgb = x <= 0.0031308 ? 12.92 * x
                                           : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
        return static_cast<unsigned char>(std::lround(255.0 * srgb));
    };
    const unsigned char padding[3]{0, 0, 0};
    const std::uint32_t paddingBytes = rowBytes - static_cast<std::uint32_t>(image.width * 3);
    for (int y = image.height - 1; y >= 0; --y) {
        for (int x = 0; x < image.width; ++x) {
            const Spectrum& pixel = image.pixels[static_cast<std::size_t>(y * image.width + x)];
            const unsigned char bgr[3]{encode(pixel.z()), encode(pixel.y()), encode(pixel.x())};
            stream.write(reinterpret_cast<const char*>(bgr), sizeof(bgr));
        }
        stream.write(reinterpret_cast<const char*>(padding), paddingBytes);
    }
}

} // namespace mf

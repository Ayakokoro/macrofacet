#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>

namespace mf {
namespace {

Spectrum environmentEmission(const Vector3& direction, const std::string& environment) {
    if (environment == "unit_white") return Spectrum::Ones();
    const double up = 0.5 * (normalizedOrThrow(direction).z() + 1.0);
    return Spectrum(0.15 + 0.85 * up, 0.2 + 0.7 * up, 0.35 + 0.55 * up);
}

FlightSample sampleTabulatedFlight(const FlightKernel& kernel, Random& rng,
                                   const NumericPolicy& policy, int cells) {
    const double a = kernel.currentAge();
    const double b = kernel.maximumAgeInDomain();
    const double target = -std::log1p(-rng.openUniform01());
    double lowerAge = a;
    double accumulatedDepth = 0.0;
    for (int i = 1; i <= cells; ++i) {
        const double upperAge = i == cells ? b : a + (b - a) * i / cells;
        const PositiveResult interval = integrateHazard(kernel, lowerAge, upperAge, policy);
        const double intervalDepth = interval.status == NumericStatus::ExactZero
            ? 0.0 : interval.value;
        const double nextDepth = accumulatedDepth + intervalDepth;
        if (target < nextDepth) {
            const double fraction = intervalDepth > 0.0
                ? (target - accumulatedDepth) / intervalDepth : 0.0;
            const double age = lowerAge + fraction * (upperAge - lowerAge);
            const double piecewiseHazard = intervalDepth / (upperAge - lowerAge);
            return {true, age, -target, std::log(piecewiseHazard) - target, std::nullopt};
        }
        accumulatedDepth = nextDepth;
        lowerAge = upperAge;
    }
    return {false, b, -accumulatedDepth, std::nullopt, std::exp(-accumulatedDepth)};
}

int resolveWorkerCount(int requested, int rows) {
    if (requested > 0) return std::max(1, std::min(requested, rows));
    const unsigned hardware = std::thread::hardware_concurrency();
    const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
    return std::max(1, std::min(available, rows));
}

} // namespace

Spectrum traceCameraPath(const Ray& initialRay, ModelMode mode,
                         const ExperimentConfig& config, Random& rng,
                         RenderStatistics& statistics) {
    const Vector3 initialDirection = normalizedOrThrow(initialRay.direction);
    const DomainInterval firstInterval = config.field.activeDomain.intersect(
        {initialRay.origin, initialDirection});
    if (!firstInterval.hit) {
        ++statistics.escapedPaths;
        return environmentEmission(initialDirection, config.render.environment);
    }
    const Point3 entry = initialRay.origin + firstInterval.entry * initialDirection;
    FlightState state = startExternalFlight(entry, initialDirection);
    if (mode == ModelMode::Conditional29 &&
        config.conditional29.externalPolicy == ExternalPolicy::SampledExterior)
        state = sampleExteriorFlight(config.field, entry, initialDirection, rng, config.numeric);
    Spectrum throughput = Spectrum::Ones();
    int depth = 0;
    for (; mode == ModelMode::Conditional29
             ? (!config.conditional29.hardDepthCap || depth < *config.conditional29.hardDepthCap)
             : depth < config.render.safetyDepthCap; ++depth) {
        try {
            std::unique_ptr<FlightKernel> kernel = makeFlightKernel(
                mode, config.field, state, config.conditional29.externalPolicy, config.numeric);
            const FlightSample flight = mode == ModelMode::Conditional29
                ? sampleFlight(*kernel, rng, config.numeric, &statistics.tracking, config.render.flightTableCells)
                : sampleTabulatedFlight(*kernel, rng, config.numeric, config.render.flightTableCells);
            if (!flight.collided) {
                ++statistics.escapedPaths;
                statistics.accumulatedPathDepth += depth;
                return throughput.cwiseProduct(
                    environmentEmission(state.direction, config.render.environment));
            }
            const Point3 hit = state.birthPosition + flight.age * state.direction;
            ++statistics.realCollisions;
            if (mode == ModelMode::Classic) {
                const double mixture = config.classicPhaseProposal == "paper_mixture"
                    ? config.beckmannMixtureWeight : 0.0;
                const bool useTargetVndf = config.classicPhaseProposal == "target_vndf";
                ConductorPhase phase(config.field, hit, mixture, useTargetVndf);
                const PhaseSample sample = phase.samplePhase(state.direction, rng);
                throughput = throughput.cwiseProduct(sample.throughputWeight);
                state = startClassicCollisionFlight(hit, sample.direction);
            } else {
                const Vector3 gradient = sampleCollisionGradient(*kernel, flight.age, rng,
                                                                 config.numeric);
                const Vector3 normal = normalizedOrThrow(gradient);
                const Vector3 outgoing = reflectTravelDirection(state.direction, normal);
                throughput = throughput.cwiseProduct(
                    conductorFresnel(-state.direction.dot(normal),
                                     config.field.conductor));
                state = startSurfaceFlight(hit, gradient, outgoing);
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
        } catch (const NumericError& error) {
            ++statistics.numericalFailures;
            statistics.accumulatedPathDepth += depth;
            if (mode == ModelMode::Conditional29) {
                std::ostringstream message;
                message << "conditional29 " << toString(error.status()) << ": " << error.what()
                        << "; depth=" << depth << "; birth=" << state.birthPosition.transpose()
                        << "; direction=" << state.direction.transpose() << "; age=" << state.age;
                throw NumericError(error.status(), message.str());
            }
            return Spectrum::Zero();
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth += depth;
    return Spectrum::Zero();
}

RenderedImage renderAnalyticScene(ModelMode mode, const ExperimentConfig& config) {
    RenderedImage result;
    result.width = config.render.width;
    result.height = config.render.height;
    result.pixels.assign(static_cast<std::size_t>(result.width * result.height), Spectrum::Zero());
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
            Random rng(config.seed + 0x9e3779b97f4a7c15ULL * (pixelIndex + 1) +
                       104729ULL * static_cast<unsigned>(mode));
            for (int sample = 0; sample < config.render.samplesPerPixel; ++sample) {
                const double px = (2.0 * ((x + rng.openUniform01()) / result.width) - 1.0) *
                                  aspect * scale;
                const double py = (1.0 - 2.0 * ((y + rng.openUniform01()) / result.height)) * scale;
                const Vector3 direction = normalizedOrThrow(forward + px * right + py * up);
                ++statistics.paths;
                sum += traceCameraPath({config.render.cameraPosition, direction}, mode,
                                       config, rng, statistics);
            }
            result.pixels[static_cast<std::size_t>(y * result.width + x)] =
                sum / config.render.samplesPerPixel;
        }
    };

    // Dynamic row assignment: cost per row varies by orders of magnitude across a
    // correlated-mode image, so static chunks would leave workers idle.
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

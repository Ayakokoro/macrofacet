// Instrumented single-thread decomposition of the neural renderer's path loop.
// This diagnostic deliberately lives outside the production sampling hot path.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: macrofacet_renewal_benchmark <render-config> <spp> <report.json>\n";
        return 2;
    }
    try {
        using namespace mf;
        registerNanoVdbFieldTypes();
        auto config = loadExperimentConfig(argv[1]);
        const int spp = std::stoi(argv[2]);
        if (spp < 1) throw std::invalid_argument("positive benchmark spp required");
        config.transportMode = "neural_renewal";
        config.renewal.backend = "scalar"; // This tool decomposes the scalar path loop.
        if (config.render.environment == "unit_white")
            config.material.conductor.forceUnitFresnel = true;
        prepareRenewalModel(config);
        prepareNanoVdbField(config);
        const auto& model = *config.renewal.model;
        if (config.renewal.profileMode != "cubic")
            throw std::invalid_argument("scalar diagnostic requires cubic; use macrofacet_profile_renewal for point_linear");
        const RenewalMedium medium(config.field, model, config.renewal.profileMaximumStep);
        const Vector3 forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
        Vector3 right, up;
        orthonormalComplement(forward, right, up);
        const double scale = std::tan(config.render.verticalFovDegrees*kPi/360);
        const double aspect = static_cast<double>(config.render.width)/config.render.height;
        std::map<std::string,double> seconds;
        std::uint64_t paths = 0, flights = 0, segments = 0, profileSegments = 0, hits = 0;
        const auto totalStart = Clock::now();
        for (int y = 0; y < config.render.height; ++y) for (int px = 0; px < config.render.width; ++px) {
            const std::uint64_t pixel = static_cast<std::uint64_t>(y*config.render.width+px);
            Random rng(config.seed+0x9e3779b97f4a7c15ULL*(pixel+1));
            for (int sample = 0; sample < spp; ++sample) {
                ++paths;
                const double u = (2*((px+rng.openUniform01())/config.render.width)-1)*aspect*scale;
                const double v = (1-2*((y+rng.openUniform01())/config.render.height))*scale;
                Ray ray{config.render.cameraPosition, normalizedOrThrow(forward+u*right+v*up)};
                const auto domain = config.field.activeDomain.intersect(ray);
                if (!domain.hit || !(domain.exit > domain.entry)) continue;
                ray.origin += domain.entry*ray.direction;
                ray.origin = ray.origin.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
                std::optional<Vector3> gradient;
                Spectrum throughput = Spectrum::Ones();
                for (int depth = 0; depth < config.render.safetyDepthCap; ++depth) {
                    auto mark = Clock::now();
                    const auto flight = medium.beginFlight(ray, gradient);
                    seconds["mean_profile"] += elapsed(mark);
                    if (!flight) break;
                    ++flights;
                    profileSegments += flight->mean.segments().size();
                    mark = Clock::now();
                    auto state = model.initialize(flight->mean.segments().front(), flight->start);
                    seconds["initial_network"] += elapsed(mark);
                    double remaining = -std::log(rng.openUniform01());
                    bool collided = false;
                    for (const auto& segment : flight->mean.segments()) {
                        ++segments;
                        mark = Clock::now();
                        auto step = model.evaluate(state, segment);
                        seconds["segment_network"] += elapsed(mark);
                        mark = Clock::now();
                        const RenewalHazardSegment hazard(segment.begin, segment.end, step.rates);
                        const double mass = hazard.integral(0,1);
                        seconds["hazard_integration"] += elapsed(mark);
                        if (mass > 0 && remaining <= mass) {
                            mark = Clock::now();
                            const double t = hazard.inverse(remaining, 0, 1, config.numeric.distanceAbsoluteTolerance/flight->ell);
                            const double x = segment.begin+(segment.end-segment.begin)*t;
                            const double distance = x*flight->ell;
                            Point3 position = ray.origin+distance*ray.direction;
                            position = position.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
                            seconds["cumulative_inversion"] += elapsed(mark);
                            mark = Clock::now();
                            const double speed = model.mixture(state, segment, t).sample(rng);
                            seconds["speed_mixture"] += elapsed(mark);
                            mark = Clock::now();
                            const Point3 interior = ray.origin+(0.5*(segment.begin+segment.end)*flight->ell)*ray.direction;
                            const Vector3 meanGradient = config.field.mean->evaluateInCell(position, interior).gradient;
                            gradient = sampleRenewalGradient(config.field.kernel, ray.direction, x,
                                -flight->sigma/flight->ell*speed, meanGradient, flight->birthMeanGradient, gradient, rng);
                            const Vector3 normal = normalizedOrThrow(*gradient);
                            seconds["gradient_normal"] += elapsed(mark);
                            throughput = throughput.cwiseProduct(conductorFresnel(-ray.direction.dot(normal), config.material.conductor));
                            ray = {position, normalizedOrThrow(reflectTravelDirection(ray.direction, normal))};
                            ++hits;
                            collided = true;
                            break;
                        }
                        remaining -= mass;
                        state = std::move(step.nextState);
                    }
                    if (!collided) break;
                    if (depth+1 >= config.render.rouletteStartDepth) {
                        const double continuation = std::clamp(throughput.maxCoeff(), 0.05, 0.95);
                        if (rng.openUniform01() >= continuation) break;
                        throughput /= continuation;
                    }
                }
            }
        }
        const double total = elapsed(totalStart);
        double measured = 0;
        for (const auto& entry : seconds) measured += entry.second;
        seconds["other_and_instrumentation"] = total-measured;
        nlohmann::json fractions;
        for (const auto& entry : seconds) fractions[entry.first] = entry.second/total;
        nlohmann::json report{{"config",argv[1]}, {"environment",config.render.environment},
            {"threads",1}, {"spp",spp}, {"paths",paths},
            {"flights",flights}, {"evaluated_segments",segments}, {"constructed_segments",profileSegments},
            {"hits",hits}, {"total_seconds",total}, {"seconds",seconds}, {"fractions",fractions},
            {"checkpoint_sha256",model.checkpointSha256()}, {"eigen_simd",Eigen::SimdInstructionSetsInUse()},
            {"note","Instrumented single-thread wall time; excludes field/model loading; timers add overhead."}};
        const std::filesystem::path output(argv[3]);
        if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("cannot write benchmark report");
        stream << report.dump(2) << '\n';
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

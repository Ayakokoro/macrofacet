#include "macrofacet/experiments/TransmittanceCurves.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mf {
namespace {

struct RayGeometry {
    bool active = false;
    double entryDistance = 0.0;
    double exitDistance = 0.0;
    double activeLength = 0.0;
    Point3 entry = Point3::Zero();
    std::string status = "miss_domain";
};

struct TrialResult {
    bool collided = false;
    double collisionDistance = 0.0;
    double mediumAge = 0.0;
};

struct CurvePoint {
    double distance = 0.0;
    double mediumDistance = 0.0;
    double transmittance = 1.0;
    double ciLow = 1.0;
    double ciHigh = 1.0;
    double exponential = 1.0;
};

struct ModeResult {
    std::string mode;
    std::vector<TrialResult> trials;
    std::vector<CurvePoint> curve;
    DdaTrackingDiagnostics diagnostics;
    int collisionCount = 0;
    double exponentialRate = 0.0;
};

struct RayResult {
    TransmittanceRayConfig ray;
    RayGeometry geometry;
    std::vector<ModeResult> modes;
};

std::ofstream openOutput(const std::filesystem::path& path) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot open output file: " + path.string());
    return stream;
}

std::string csv(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result = "\"";
    for (char c : value) result += c == '\"' ? "\"\"" : std::string(1, c);
    return result + '"';
}

std::string safeFileComponent(const std::string& value) {
    std::string result;
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_') {
            result += static_cast<char>(c);
        } else {
            result += '_';
        }
    }
    return result.empty() ? "ray" : result;
}

std::string xml(const std::string& value) {
    std::string result;
    for (char c : value) {
        if (c == '&') result += "&amp;";
        else if (c == '<') result += "&lt;";
        else if (c == '>') result += "&gt;";
        else if (c == '\"') result += "&quot;";
        else if (c == '\'') result += "&apos;";
        else result += c;
    }
    return result;
}

std::uint64_t fnv1a(const std::string& text) {
    std::uint64_t value = 14695981039346656037ULL;
    for (unsigned char c : text) {
        value ^= c;
        value *= 1099511628211ULL;
    }
    return value;
}

std::uint64_t trialSeed(std::uint64_t base, const std::string& rayId,
                        const std::string& mode, int trial) {
    std::uint64_t value = base ^ fnv1a(rayId + "|" + mode);
    value ^= 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(trial) +
             (value << 6U) + (value >> 2U);
    return value;
}

RayGeometry clipRay(const Bounds3& domain, const TransmittanceRayConfig& ray) {
    RayGeometry result;
    const DomainInterval interval = domain.intersect({ray.origin, ray.direction});
    if (!interval.hit || !(interval.exit > interval.entry) ||
        !(ray.maximumDistance > interval.entry)) {
        return result;
    }
    result.entryDistance = interval.entry;
    result.exitDistance = std::min(interval.exit, ray.maximumDistance);
    if (!(result.exitDistance > result.entryDistance)) return result;
    result.active = true;
    result.activeLength = result.exitDistance - result.entryDistance;
    result.entry = ray.origin + result.entryDistance * ray.direction;
    result.status = "active";
    return result;
}

std::pair<double, double> wilsonInterval(int successes, int trials, double z) {
    if (trials <= 0) return {1.0, 1.0};
    const double n = static_cast<double>(trials);
    const double p = successes / n;
    const double z2 = z * z;
    const double denominator = 1.0 + z2 / n;
    const double center = (p + 0.5 * z2 / n) / denominator;
    const double radius = z * std::sqrt((p * (1.0 - p) + 0.25 * z2 / n) / n) /
                          denominator;
    return {std::max(0.0, center - radius), std::min(1.0, center + radius)};
}

void runLegacyCurves(const ExperimentConfig& config, int rayCount, int bins) {
    if (rayCount < 1 || bins < 2) throw std::invalid_argument("invalid legacy curve budget");
    const std::array<const char*, 3> names{
        "classic_local", "classic_global", "global_conditional"};
    std::array<std::vector<double>, 3> survival;
    for (auto& values : survival) values.assign(static_cast<std::size_t>(bins + 1), 0.0);
    std::array<int, 3> traced{0, 0, 0};
    std::ofstream individual = openOutput(config.outputDirectory / "transmittance_rays.csv");
    individual << "ray_id,mode,origin_x,origin_y,origin_z,dir_x,dir_y,dir_z,"
                  "entry_x,entry_y,entry_z,domain_distance,collision_distance,escaped\n"
               << std::setprecision(17);
    const Vector3 direction = normalizedOrThrow(config.render.cameraTarget -
                                                 config.render.cameraPosition);
    Vector3 right, up;
    orthonormalComplement(direction, right, up);
    const Vector3 extent = config.field.activeDomain.maximum -
                           config.field.activeDomain.minimum;
    const double maximumDistance = extent.cwiseProduct(direction.cwiseAbs()).sum();
    const double span = 0.75 * extent.maxCoeff();
    MaterialConfig localMaterial = config.material;
    localMaterial.gpModel = GpModel::LocalTangent;
    MaterialConfig globalMaterial = config.material;
    globalMaterial.gpModel = GpModel::GlobalPointwise;
    if (globalMaterial.ndfFamily != NdfFamily::GeneralizedGaussian ||
        globalMaterial.alphaField) {
        throw std::invalid_argument(
            "legacy conditional curves require generalized_gaussian and "
            "field.use_alpha_grid=false");
    }
    const NarrowBandMedium localMedium(config.field, localMaterial, config.mediumDensity,
        config.densityMajorantGrid,
        config.material.gpModel == GpModel::LocalTangent
            ? config.preparedAreaMajorant : std::nullopt);
    const NarrowBandMedium globalMedium(config.field, globalMaterial, config.mediumDensity,
        config.densityMajorantGrid,
        config.material.gpModel == GpModel::GlobalPointwise
            ? config.preparedAreaMajorant : std::nullopt);
    const ConditionalMedium conditionalMedium(config.field);
    const int nx = static_cast<int>(std::ceil(std::sqrt(rayCount)));
    const int ny = (rayCount + nx - 1) / nx;
    for (int i = 0; i < rayCount; ++i) {
        const double u = 2.0 * ((i % nx + 0.5) / nx) - 1.0;
        const double v = 2.0 * ((i / nx + 0.5) / ny) - 1.0;
        const Point3 origin = config.render.cameraPosition + span * (u * right + v * up);
        const DomainInterval interval = config.field.activeDomain.intersect({origin, direction});
        if (!interval.hit || !(interval.exit > interval.entry)) continue;
        const Point3 entry = origin + interval.entry * direction;
        const double domainDistance = interval.exit - interval.entry;
        for (int mode = 0; mode < 3; ++mode) {
            Random rng(config.seed + 0x9e3779b97f4a7c15ULL *
                       (static_cast<std::uint64_t>(i) + 1) + mode * 0x100000001b3ULL);
            FlightSample sample;
            if (mode == 2) {
                const FlightState state = conditionalMedium.startExternal(entry, direction, rng);
                const ConditionalFlightKernel kernel = conditionalMedium.beginFlight(state);
                sample = conditionalMedium.sample(kernel, rng);
            } else {
                const NarrowBandMedium& medium = mode == 0 ? localMedium : globalMedium;
                const ClassicFlightKernel kernel = medium.beginFlight(
                    medium.startExternal(entry, direction));
                sample = medium.sample(kernel, rng, config.numeric);
            }
            const double collision = sample.collided ? sample.age : domainDistance;
            ++traced[mode];
            for (int bin = 0; bin <= bins; ++bin) {
                const double distance = maximumDistance * bin / bins;
                if (!sample.collided || collision > distance)
                    survival[mode][static_cast<std::size_t>(bin)] += 1.0;
            }
            individual << i << ',' << names[mode] << ',' << origin.x() << ','
                << origin.y() << ',' << origin.z() << ',' << direction.x() << ','
                << direction.y() << ',' << direction.z() << ',' << entry.x() << ','
                << entry.y() << ',' << entry.z() << ',' << domainDistance << ','
                << collision << ',' << (!sample.collided ? 1 : 0) << '\n';
        }
    }
    std::ofstream curve = openOutput(config.outputDirectory / "transmittance_curves.csv");
    curve << "distance,classic_local,classic_global,global_conditional,ray_count\n"
          << std::setprecision(17);
    for (int bin = 0; bin <= bins; ++bin) {
        curve << maximumDistance * bin / bins;
        for (int mode = 0; mode < 3; ++mode)
            curve << ',' << (traced[mode] > 0
                ? survival[mode][static_cast<std::size_t>(bin)] / traced[mode] : 1.0);
        curve << ',' << traced[0] << '\n';
    }
    std::ofstream svg = openOutput(config.outputDirectory / "transmittance_curves.svg");
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 720 440\">"
           "<rect width=\"720\" height=\"440\" fill=\"white\"/>"
           "<path d=\"M70 30 V370 H690\" fill=\"none\" stroke=\"black\"/>"
           "<text x=\"300\" y=\"420\">distance from domain entry</text>"
           "<text x=\"12\" y=\"25\">survival</text>"
           "<text x=\"48\" y=\"374\">0</text>"
           "<text x=\"48\" y=\"35\">1</text>";
    const std::array<const char*, 3> colors{"#147d99", "#e47824", "#783a9a"};
    for (int mode = 0; mode < 3; ++mode) {
        svg << "<polyline fill=\"none\" stroke=\"" << colors[mode]
            << "\" stroke-width=\"2\" points=\"";
        for (int bin = 0; bin <= bins; ++bin) {
            const double value = traced[mode] > 0
                ? survival[mode][static_cast<std::size_t>(bin)] / traced[mode] : 1.0;
            svg << 70.0 + 620.0 * bin / bins << ',' << 370.0 - 340.0 * value << ' ';
        }
        svg << "\"/><text x=\"510\" y=\"" << 58 + 22 * mode
            << "\" fill=\"" << colors[mode] << "\">" << names[mode] << "</text>";
    }
    svg << "</svg>\n";
}

void makeCurve(ModeResult& result, const TransmittanceRayConfig& ray,
               const RayGeometry& geometry, int bins, double confidence,
               bool exponentialEnabled) {
    std::vector<double> collisions;
    collisions.reserve(result.trials.size());
    double exposure = 0.0;
    for (const TrialResult& trial : result.trials) {
        if (trial.collided) {
            collisions.push_back(trial.collisionDistance);
            exposure += trial.mediumAge;
        } else {
            exposure += geometry.activeLength;
        }
    }
    std::sort(collisions.begin(), collisions.end());
    result.collisionCount = static_cast<int>(collisions.size());
    result.exponentialRate = exposure > 0.0
        ? result.collisionCount / exposure : 0.0;
    const int trialCount = static_cast<int>(result.trials.size());
    const double z = normalQuantile(0.5 + 0.5 * confidence);
    result.curve.reserve(static_cast<std::size_t>(bins + 1));
    for (int bin = 0; bin <= bins; ++bin) {
        CurvePoint point;
        point.distance = ray.maximumDistance * bin / bins;
        point.mediumDistance = geometry.active
            ? std::clamp(point.distance - geometry.entryDistance, 0.0,
                         geometry.activeLength)
            : 0.0;
        const int failed = static_cast<int>(std::upper_bound(
            collisions.begin(), collisions.end(), point.distance) - collisions.begin());
        const int surviving = trialCount - failed;
        point.transmittance = trialCount > 0
            ? static_cast<double>(surviving) / trialCount : 1.0;
        if (!geometry.active || point.mediumDistance == 0.0) {
            point.ciLow = point.ciHigh = 1.0;
        } else {
            const auto interval = wilsonInterval(surviving, trialCount, z);
            point.ciLow = interval.first;
            point.ciHigh = interval.second;
        }
        point.exponential = exponentialEnabled
            ? std::exp(-result.exponentialRate * point.mediumDistance)
            : 1.0;
        result.curve.push_back(point);
    }
}

const ModeResult* findMode(const RayResult& ray, const std::string& mode) {
    const auto found = std::find_if(ray.modes.begin(), ray.modes.end(),
        [&](const ModeResult& value) { return value.mode == mode; });
    return found == ray.modes.end() ? nullptr : &*found;
}

void writeRaySvg(const RayResult& result, const TransmittanceConfig& settings,
                 const std::filesystem::path& path) {
    std::ofstream svg = openOutput(path);
    svg << std::setprecision(17)
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 760 460\">"
           "<rect width=\"760\" height=\"460\" fill=\"white\"/>"
           "<path d=\"M70 35 V385 H710\" fill=\"none\" stroke=\"black\"/>"
           "<text x=\"300\" y=\"435\">distance from requested ray origin</text>"
           "<text x=\"12\" y=\"25\">transmittance</text>"
           "<text x=\"48\" y=\"389\">0</text>"
           "<text x=\"48\" y=\"40\">1</text>";
    const std::vector<std::string> colors{"#147d99", "#e47824", "#783a9a"};
    for (std::size_t mode = 0; mode < result.modes.size(); ++mode) {
        const ModeResult& values = result.modes[mode];
        const std::string& color = colors[mode % colors.size()];
        svg << "<polyline fill=\"none\" stroke=\"" << color
            << "\" stroke-width=\"2\" points=\"";
        for (const CurvePoint& point : values.curve) {
            svg << 70.0 + 640.0 * point.distance / result.ray.maximumDistance << ','
                << 385.0 - 350.0 * point.transmittance << ' ';
        }
        svg << "\"/>";
        if (settings.constantExponentialEnabled) {
            svg << "<polyline fill=\"none\" stroke=\"" << color
                << "\" stroke-width=\"1.5\" stroke-dasharray=\"7 5\" opacity=\"0.75\" points=\"";
            for (const CurvePoint& point : values.curve) {
                svg << 70.0 + 640.0 * point.distance / result.ray.maximumDistance << ','
                    << 385.0 - 350.0 * point.exponential << ' ';
            }
            svg << "\"/>";
        }
        svg << "<text x=\"455\" y=\"" << 60 + 22 * mode << "\" fill=\""
            << color << "\">" << values.mode << " (solid), exp fit (dash)</text>";
    }
    if (result.geometry.active) {
        const double x = 70.0 + 640.0 * result.geometry.entryDistance /
                                  result.ray.maximumDistance;
        svg << "<path d=\"M" << x << " 35 V385\" stroke=\"#777\" "
               "stroke-dasharray=\"2 4\"/><text x=\"" << x + 4
            << "\" y=\"378\" fill=\"#555\">medium entry</text>";
    }
    svg << "<text x=\"70\" y=\"415\">ray: " << xml(result.ray.id)
        << "</text></svg>\n";
}

} // namespace

void runTransmittanceCurves(const ExperimentConfig& config, int legacyRayCount,
                            int binsOverride) {
    requireNanoVdbField(config);
    std::filesystem::create_directories(config.outputDirectory);
    if (!config.transmittance) {
        runLegacyCurves(config, legacyRayCount, binsOverride);
        return;
    }
    TransmittanceConfig settings = *config.transmittance;
    if (settings.rays.empty()) throw std::invalid_argument("transmittance ray list is empty");

    const bool needsLocal = std::find(settings.modes.begin(), settings.modes.end(),
        "classic_local") != settings.modes.end();
    const bool needsGlobal = std::find(settings.modes.begin(), settings.modes.end(),
        "classic_global") != settings.modes.end();
    const bool needsConditional = std::find(settings.modes.begin(), settings.modes.end(),
        "global_conditional") != settings.modes.end();

    MaterialConfig localMaterial = config.material;
    localMaterial.gpModel = GpModel::LocalTangent;
    MaterialConfig globalMaterial = config.material;
    globalMaterial.gpModel = GpModel::GlobalPointwise;
    if (needsConditional && (globalMaterial.ndfFamily != NdfFamily::GeneralizedGaussian ||
                             globalMaterial.alphaField)) {
        throw std::invalid_argument(
            "global_conditional transmittance requires generalized_gaussian and "
            "field.use_alpha_grid=false");
    }

    std::unique_ptr<NarrowBandMedium> localMedium;
    std::unique_ptr<NarrowBandMedium> globalMedium;
    if (needsLocal) {
        localMedium = std::make_unique<NarrowBandMedium>(
            config.field, localMaterial, config.mediumDensity, config.densityMajorantGrid,
            config.material.gpModel == GpModel::LocalTangent
                ? config.preparedAreaMajorant : std::nullopt);
    }
    if (needsGlobal) {
        globalMedium = std::make_unique<NarrowBandMedium>(
            config.field, globalMaterial, config.mediumDensity, config.densityMajorantGrid,
            config.material.gpModel == GpModel::GlobalPointwise
                ? config.preparedAreaMajorant : std::nullopt);
    }
    const ConditionalMedium conditionalMedium(config.field);

    std::optional<std::ofstream> raw;
    if (settings.writeRawSamples) {
        raw.emplace(openOutput(config.outputDirectory / "transmittance_samples.csv"));
        *raw << "ray_id,group,mode,trial,status,collided,collision_distance,"
                "medium_age,medium_entry_distance,medium_exit_distance\n"
             << std::setprecision(17);
    }

    std::vector<RayResult> results;
    results.reserve(settings.rays.size());
    for (const TransmittanceRayConfig& ray : settings.rays) {
        RayResult rayResult;
        rayResult.ray = ray;
        rayResult.geometry = clipRay(config.field.activeDomain, ray);
        const ConditionalBirthConfig& birth = ray.conditionalBirth
            ? *ray.conditionalBirth : settings.conditionalBirth;
        for (const std::string& mode : settings.modes) {
            ModeResult modeResult;
            modeResult.mode = mode;
            modeResult.trials.reserve(static_cast<std::size_t>(settings.trialsPerRay));
            for (int trial = 0; trial < settings.trialsPerRay; ++trial) {
                TrialResult observed;
                if (rayResult.geometry.active) {
                    try {
                        Random rng(trialSeed(config.seed, ray.id, mode, trial));
                        FlightSample sample;
                        if (mode == "global_conditional") {
                            FlightState state;
                            if (birth.policy == "fixed_observation") {
                                state = startExternalFlight(rayResult.geometry.entry, ray.direction);
                                state.birthValue = birth.value;
                                state.birthGradient = birth.gradient;
                            } else {
                                state = conditionalMedium.startExternal(
                                    rayResult.geometry.entry, ray.direction, rng);
                            }
                            const ConditionalFlightKernel flight =
                                conditionalMedium.beginFlight(state);
                            sample = conditionalMedium.sample(
                                flight, rng, &modeResult.diagnostics,
                                rayResult.geometry.activeLength);
                        } else {
                            const NarrowBandMedium& medium = mode == "classic_local"
                                ? *localMedium : *globalMedium;
                            const ClassicFlightKernel flight = medium.beginFlight(
                                medium.startExternal(rayResult.geometry.entry, ray.direction));
                            sample = medium.sample(flight, rng, config.numeric,
                                                   &modeResult.diagnostics,
                                                   rayResult.geometry.activeLength);
                        }
                        observed.collided = sample.collided;
                        observed.mediumAge = sample.age;
                        observed.collisionDistance = rayResult.geometry.entryDistance + sample.age;
                    } catch (const std::exception& error) {
                        throw std::runtime_error(
                            "transmittance ray '" + ray.id + "', mode " + mode +
                            ", trial " + std::to_string(trial) + ": " + error.what());
                    }
                }
                modeResult.trials.push_back(observed);
                if (raw) {
                    *raw << csv(ray.id) << ',' << csv(ray.group) << ',' << mode << ','
                         << trial << ',' << rayResult.geometry.status << ','
                         << (observed.collided ? 1 : 0) << ',';
                    if (observed.collided) *raw << observed.collisionDistance;
                    *raw << ',' << observed.mediumAge << ','
                         << rayResult.geometry.entryDistance << ','
                         << rayResult.geometry.exitDistance << '\n';
                }
            }
            makeCurve(modeResult, ray, rayResult.geometry,
                      ray.bins.value_or(settings.bins), settings.confidenceLevel,
                      settings.constantExponentialEnabled);
            rayResult.modes.push_back(std::move(modeResult));
        }
        results.push_back(std::move(rayResult));
    }

    std::ofstream rays = openOutput(config.outputDirectory / "transmittance_rays.csv");
    rays << "ray_id,group,status,origin_x,origin_y,origin_z,dir_x,dir_y,dir_z,"
            "requested_distance,medium_entry_distance,medium_exit_distance,medium_length\n"
         << std::setprecision(17);
    for (const RayResult& result : results) {
        rays << csv(result.ray.id) << ',' << csv(result.ray.group) << ','
             << result.geometry.status << ',' << result.ray.origin.x() << ','
             << result.ray.origin.y() << ',' << result.ray.origin.z() << ','
             << result.ray.direction.x() << ',' << result.ray.direction.y() << ','
             << result.ray.direction.z() << ',' << result.ray.maximumDistance << ','
             << result.geometry.entryDistance << ',' << result.geometry.exitDistance << ','
             << result.geometry.activeLength << '\n';
    }

    std::ofstream curves = openOutput(config.outputDirectory / "transmittance_curves.csv");
    curves << "ray_id,group,status,mode,distance,distance_in_medium,transmittance,"
              "ci_low,ci_high,classic_global_reference,constant_exponential,"
              "delta_to_classic_global,delta_to_constant_exponential,trials\n"
           << std::setprecision(17);
    for (const RayResult& result : results) {
        const ModeResult* reference = findMode(result, settings.localTransportReference);
        for (const ModeResult& mode : result.modes) {
            for (std::size_t index = 0; index < mode.curve.size(); ++index) {
                const CurvePoint& point = mode.curve[index];
                const bool hasReference = reference && reference->curve.size() == mode.curve.size();
                curves << csv(result.ray.id) << ',' << csv(result.ray.group) << ','
                       << result.geometry.status << ',' << mode.mode << ','
                       << point.distance << ',' << point.mediumDistance << ','
                       << point.transmittance << ',' << point.ciLow << ',' << point.ciHigh << ',';
                if (hasReference) curves << reference->curve[index].transmittance;
                curves << ',';
                if (settings.constantExponentialEnabled) curves << point.exponential;
                curves << ',';
                if (hasReference)
                    curves << point.transmittance - reference->curve[index].transmittance;
                curves << ',';
                if (settings.constantExponentialEnabled)
                    curves << point.transmittance - point.exponential;
                curves << ',' << mode.trials.size() << '\n';
            }
        }
    }

    std::ofstream summary = openOutput(config.outputDirectory / "transmittance_summary.csv");
    summary << "ray_id,group,status,mode,trials,collisions,escapes,endpoint_transmittance,"
               "constant_exponential_rate,max_abs_delta_to_classic_global,"
               "max_abs_delta_to_constant_exponential,mean_abs_delta_to_classic_global,"
               "mean_abs_delta_to_constant_exponential,candidates,null_collisions,"
               "bound_intervals,near_candidates,far_candidates\n"
            << std::setprecision(17);
    for (const RayResult& result : results) {
        const ModeResult* reference = findMode(result, settings.localTransportReference);
        for (const ModeResult& mode : result.modes) {
            double maxReference = 0.0, maxExponential = 0.0;
            double sumReference = 0.0, sumExponential = 0.0;
            const bool hasReference = reference && reference->curve.size() == mode.curve.size();
            for (std::size_t index = 0; index < mode.curve.size(); ++index) {
                if (hasReference) {
                    const double difference = std::abs(
                        mode.curve[index].transmittance -
                        reference->curve[index].transmittance);
                    maxReference = std::max(maxReference, difference);
                    sumReference += difference;
                }
                if (settings.constantExponentialEnabled) {
                    const double difference = std::abs(
                        mode.curve[index].transmittance - mode.curve[index].exponential);
                    maxExponential = std::max(maxExponential, difference);
                    sumExponential += difference;
                }
            }
            const double count = static_cast<double>(mode.curve.size());
            summary << csv(result.ray.id) << ',' << csv(result.ray.group) << ','
                    << result.geometry.status << ',' << mode.mode << ','
                    << mode.trials.size() << ',' << mode.collisionCount << ','
                    << mode.trials.size() - static_cast<std::size_t>(mode.collisionCount) << ','
                    << mode.curve.back().transmittance << ',' << mode.exponentialRate << ',';
            if (hasReference) summary << maxReference;
            summary << ',';
            if (settings.constantExponentialEnabled) summary << maxExponential;
            summary << ',';
            if (hasReference) summary << sumReference / count;
            summary << ',';
            if (settings.constantExponentialEnabled) summary << sumExponential / count;
            summary << ',' << mode.diagnostics.candidates << ','
                    << mode.diagnostics.nullCollisions << ','
                    << mode.diagnostics.boundIntervals << ','
                    << mode.diagnostics.nearCandidates << ','
                    << mode.diagnostics.farCandidates << '\n';
        }
        writeRaySvg(result, settings, config.outputDirectory /
            ("transmittance_" + safeFileComponent(result.ray.id) + ".svg"));
    }
}

} // namespace mf

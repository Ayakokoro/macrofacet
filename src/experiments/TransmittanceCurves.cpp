#include "macrofacet/experiments/TransmittanceCurves.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace mf {

void runTransmittanceCurves(const ExperimentConfig& config, int rayCount, int bins) {
    if (rayCount < 1 || bins < 2) throw std::invalid_argument("invalid curve budget");
    requireNanoVdbField(config);
    std::filesystem::create_directories(config.outputDirectory);
    const std::array<const char*, 3> names{
        "classic_local", "classic_global", "global_conditional"};
    std::array<std::vector<double>, 3> survival;
    for (auto& values : survival) values.assign(static_cast<std::size_t>(bins + 1), 0.0);
    std::array<int, 3> traced{0, 0, 0};
    std::ofstream individual(config.outputDirectory / "transmittance_rays.csv");
    individual << "ray_id,mode,origin_x,origin_y,origin_z,dir_x,dir_y,dir_z,"
                  "entry_x,entry_y,entry_z,domain_distance,collision_distance,escaped\n";
    individual << std::setprecision(17);
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
        globalMaterial.alphaField)
        throw std::invalid_argument(
            "conditional curves require global Gaussian NDF without alpha grid");
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
        const Point3 origin = config.render.cameraPosition +
            span * (u * right + v * up);
        const DomainInterval interval = config.field.activeDomain.intersect({origin, direction});
        if (!interval.hit || !(interval.exit > interval.entry)) continue;
        const Point3 entry = origin + interval.entry * direction;
        const double domainDistance = interval.exit - interval.entry;
        for (int mode = 0; mode < 3; ++mode) {
            Random rng(config.seed + 0x9e3779b97f4a7c15ULL *
                       (static_cast<std::uint64_t>(i) + 1) + mode * 0x100000001b3ULL);
            FlightSample sample;
            if (mode == 2) {
                const FlightState state = conditionalMedium.startExternal(entry,direction,rng);
                const ConditionalFlightKernel kernel = conditionalMedium.beginFlight(state);
                sample = conditionalMedium.sample(kernel,rng);
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
    std::ofstream curve(config.outputDirectory / "transmittance_curves.csv");
    curve << "distance,classic_local,classic_global,global_conditional,ray_count\n"
          << std::setprecision(17);
    for (int bin = 0; bin <= bins; ++bin) {
        curve << maximumDistance * bin / bins;
        for (int mode = 0; mode < 3; ++mode)
            curve << ',' << (traced[mode] > 0 ?
                survival[mode][static_cast<std::size_t>(bin)] / traced[mode] : 1.0);
        curve << ',' << traced[0] << '\n';
    }
    std::ofstream svg(config.outputDirectory / "transmittance_curves.svg");
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
            const double value = traced[mode] > 0 ?
                survival[mode][static_cast<std::size_t>(bin)] / traced[mode] : 1.0;
            svg << 70.0 + 620.0 * bin / bins << ',' << 370.0 - 340.0 * value << ' ';
        }
        svg << "\"/><text x=\"510\" y=\"" << 58 + 22 * mode
            << "\" fill=\"" << colors[mode] << "\">" << names[mode]
            << "</text>";
    }
    svg << "</svg>\n";
}

} // namespace mf

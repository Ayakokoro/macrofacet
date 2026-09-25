#include "macrofacet/fields/PrepareNanoVdbField.h"

#include "fieldgen/NanoVdbIO.h"
#include "fieldgen/VdbBaker.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/NanoVdbSampledField.h"

#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mf {
namespace {

std::uint64_t fnv1a(const std::string& text) {
    std::uint64_t value = 14695981039346656037ULL;
    for (unsigned char byte : text) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

} // namespace

void prepareNanoVdbField(ExperimentConfig& config) {
    if (!config.field.mean) throw std::invalid_argument("experiment has no mean field");
    if (config.field.mean->typeName() == std::string("nanovdb")) {
        const auto* baked = dynamic_cast<const NanoVdbMean*>(config.field.mean.get());
        if (!baked) throw std::invalid_argument("nanovdb mean has an unexpected implementation");
        const auto density = NanoVdbSampledField::open(baked->gridFile(), "density");
        config.mediumDensity = density;
        if (!density->gridBounds().valid())
            throw std::invalid_argument("nanovdb density grid has no valid spatial bounds");
        const Vector3 margin = Vector3::Constant(density->voxelSize());
        config.field.activeDomain.minimum = config.field.activeDomain.minimum.cwiseMin(
            density->gridBounds().minimum - margin);
        config.field.activeDomain.maximum = config.field.activeDomain.maximum.cwiseMax(
            density->gridBounds().maximum + margin);
        config.field.validate();
        const auto sidecar = readSidecar(baked->gridFile());
        config.mediumSurfaceBand = !sidecar || !sidecar->fullDomain;
        if (config.mediumSurfaceBand) {
            config.densityMajorantGrid = std::make_shared<DensityMajorantGrid>(
                *config.mediumDensity, config.field.activeDomain);
        }
        return;
    }
    if (config.sourceFieldSpec.empty()) {
        throw std::invalid_argument("analytic field has no source description for an NVDB bake");
    }
    const double sigma = config.field.kernel.sigma();
    double dx = config.bakeVoxelSize.value_or(sigma * 0.5);
    const double requestedDx = dx;
    if (!(dx > 0.0)) throw std::invalid_argument("invalid analytic bake voxel size");
    if (!config.bakeVoxelSize) {
        const Vector3 extent = config.field.activeDomain.maximum -
                               config.field.activeDomain.minimum;
        for (int attempt = 0; attempt < 128; ++attempt) {
            const double voxelCount = (std::ceil(extent.x() / dx) + 3.0) *
                                      (std::ceil(extent.y() / dx) + 3.0) *
                                      (std::ceil(extent.z() / dx) + 3.0);
            if (voxelCount <= 8000000.0) break;
            dx *= 1.1;
        }
    }
    if (dx > requestedDx) {
        std::cerr << "analytic NVDB bake voxel size increased from " << requestedDx
                  << " to " << dx << " to fit the 8-million-voxel limit; "
                  << "set field.bake_voxel_size explicitly to control resolution\n";
    }

    std::ostringstream signature;
    signature << "full-domain-v2|" << config.sourceFieldSpec << '|'
              << std::setprecision(17) << sigma << '|' << dx << '|'
              << config.field.activeDomain.minimum.transpose() << '|'
              << config.field.activeDomain.maximum.transpose() << '|'
              << config.material.ggxAlpha.x();
    std::ostringstream fileName;
    fileName << "analytic_" << std::hex << fnv1a(signature.str()) << ".nvdb";
    const std::filesystem::path directory = config.outputDirectory / "fields";
    const std::filesystem::path path = directory / fileName.str();
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(directory);
        const double alpha = config.material.ggxAlpha.x();
        const BakedGrids grids = bakeAnalyticMean(*config.field.mean,
            config.field.activeDomain, dx, sigma, alpha);
        BakeSettings settings;
        settings.sigma = sigma;
        settings.alpha = alpha;
        settings.resolution = {
            static_cast<int>(grids.report.voxels[0]),
            static_cast<int>(grids.report.voxels[1]),
            static_cast<int>(grids.report.voxels[2])};
        settings.bandSigmas = 0.0;  // full active domain, not a surface band
        settings.fullDomain = true;
        writeFieldFile(path, grids, settings, "<analytic mean>");
    }
    config.field.mean = NanoVdbMean::open(path, sigma);
    config.mediumDensity = NanoVdbSampledField::open(path, "density");
    config.mediumSurfaceBand = false;
    config.densityMajorantGrid.reset();
    // Procedural fields have no alpha grid. Preserve their existing material
    // NDF instead of silently replacing it with the file's constant alpha.
    config.field.validate();
}

} // namespace mf

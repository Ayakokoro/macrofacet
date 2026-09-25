#include "fieldgen/NanoVdbIO.h"

#include <nanovdb/io/IO.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace mf {
namespace {

std::filesystem::path sidecarPath(const std::filesystem::path& path) {
    return path.string() + ".json";
}

nlohmann::json toArray(const Point3& v) { return {v.x(), v.y(), v.z()}; }

Point3 vector3(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) throw std::invalid_argument("expected a 3-vector");
    return Point3(value[0].get<double>(), value[1].get<double>(), value[2].get<double>());
}

} // namespace

void writeFieldFile(const std::filesystem::path& path, const BakedGrids& grids,
                    const BakeSettings& settings, const std::string& meshName,
                    const MeshTransform& transform) {
    for (const auto& handle : grids.grids) {
        if (!handle.gridData()) throw std::runtime_error("refusing to write an empty grid handle");
    }

    // VecT is a template-template parameter with a default, so it is never
    // deduced -- both parameters must be spelled out or this fails to resolve
    // with a C2672 that points nowhere useful.
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(
        path.string(), grids.grids, nanovdb::io::Codec::NONE, 1);

    const BakeReport& report = grids.report;
    FieldSidecar sidecar;
    sidecar.sigma = settings.sigma;
    sidecar.alpha = settings.alpha;
    sidecar.dx = report.dx;
    sidecar.origin = report.origin;
    sidecar.band = settings.bandSigmas;
    sidecar.fullDomain = settings.fullDomain;
    sidecar.sdfBackground = report.background;
    sidecar.signMode = settings.signMode == SignMode::Threshold ? "threshold" : "winding";
    sidecar.grids = {"density", "alpha", "sdf", "sigma"};
    sidecar.resolution = settings.resolution;
    sidecar.voxels = report.voxels;
    sidecar.meshName = meshName;
    sidecar.maximumGradientNorm = report.maximumGradientNorm;
    sidecar.sdfMinimum = report.sdfMinimum;
    sidecar.sdfMaximum = report.sdfMaximum;

    const nlohmann::json document = {
        {"version", sidecar.version},
        {"sigma", sidecar.sigma},
        {"alpha", sidecar.alpha},
        {"dx", sidecar.dx},
        {"origin", toArray(sidecar.origin)},
        {"band", sidecar.band},
        {"coverage", sidecar.fullDomain ? "full_domain" : "surface_band"},
        {"grids", sidecar.grids},
        {"sign_mode", sidecar.signMode},
        {"resolution", {sidecar.resolution[0], sidecar.resolution[1], sidecar.resolution[2]}},
        {"voxels", {sidecar.voxels[0], sidecar.voxels[1], sidecar.voxels[2]}},
        {"mesh", sidecar.meshName},
        {"transform", {{"scale", transform.scale}, {"translation", toArray(transform.translation)}}},
        {"background", {{"sdf", report.background}, {"alpha", settings.alpha}, {"density", 0.0}}},
        {"measured", {{"sdf_min", report.sdfMinimum},
                      {"sdf_max", report.sdfMaximum},
                      {"density_max", report.densityMaximum},
                      {"maximum_gradient_norm", report.maximumGradientNorm},
                      {"band_voxels", report.bandVoxels}}},
        {"note", "maximum_gradient_norm is an upper bound, not a measurement; "
                 "it is sqrt(3) * (max value - min value) / dx for an isotropic "
                 "trilinear grid"}};

    std::ofstream stream(sidecarPath(path));
    if (!stream) throw std::runtime_error("cannot write sidecar: " + sidecarPath(path).string());
    stream << std::setw(2) << document << '\n';
}

std::optional<double> readSigmaGrid(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("cannot open field file: " + path.string());
    }
    try {
        auto handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(path.string(), "sigma");
        const auto* grid = handle.grid<float>();
        if (!grid) return std::nullopt;
        return static_cast<double>(grid->tree().root().background());
    } catch (const std::exception&) {
        // No grid by that name in this file. Absence is not an error here; the
        // caller decides whether the sidecar can supply sigma instead.
        return std::nullopt;
    }
}

std::optional<FieldSidecar> readSidecar(const std::filesystem::path& path) {
    const std::filesystem::path sidecar = sidecarPath(path);
    if (!std::filesystem::exists(sidecar)) return std::nullopt;
    std::ifstream stream(sidecar);
    if (!stream) throw std::runtime_error("cannot open sidecar: " + sidecar.string());
    nlohmann::json root;
    try {
        stream >> root;
    } catch (const std::exception& error) {
        throw std::runtime_error("malformed sidecar JSON in " + sidecar.string() + ": " + error.what());
    }
    FieldSidecar result;
    result.version = root.value("version", 1);
    result.sigma = root.value("sigma", 0.0);
    result.alpha = root.value("alpha", 0.0);
    result.dx = root.value("dx", 0.0);
    result.band = root.value("band", 0.0);
    result.fullDomain = root.value("coverage", "surface_band") == "full_domain";
    result.signMode = root.value("sign_mode", "winding");
    result.meshName = root.value("mesh", std::string());
    result.maximumGradientNorm = root.value("measured", nlohmann::json::object())
                                     .value("maximum_gradient_norm", 0.0);
    result.sdfMinimum = root.value("measured", nlohmann::json::object()).value("sdf_min", 0.0);
    result.sdfMaximum = root.value("measured", nlohmann::json::object()).value("sdf_max", 0.0);
    if (root.contains("origin")) result.origin = vector3(root["origin"]);
    if (root.contains("background")) {
        result.sdfBackground = root["background"].value("sdf", 0.0);
    }
    if (root.contains("grids")) result.grids = root["grids"].get<std::vector<std::string>>();
    if (root.contains("resolution") && root["resolution"].is_array() &&
        root["resolution"].size() == 3) {
        for (int axis = 0; axis < 3; ++axis) {
            result.resolution[axis] = root["resolution"][axis].get<int>();
        }
    }
    return result;
}

bool sameSigma(double a, double b) {
    // One float32 ulp apart is the same sigma; anything further is a re-bake.
    // NaN compares unequal to everything, including itself, which is what a
    // caller who has a non-finite sigma wants to hear.
    return static_cast<float>(a) == static_cast<float>(b);
}

double resolveSigma(const std::filesystem::path& path) {
    const std::optional<double> fromFile = readSigmaGrid(path);
    const std::optional<FieldSidecar> sidecar = readSidecar(path);
    const std::optional<double> fromJson =
        sidecar && sidecar->sigma > 0.0 ? sidecar->sigma : std::optional<double>();

    if (fromFile && fromJson) {
        if (!sameSigma(*fromFile, *fromJson)) {
            throw std::runtime_error(
                "sigma disagrees between the .nvdb (" + std::to_string(*fromFile) +
                ") and its sidecar (" + std::to_string(*fromJson) + ") for " + path.string() +
                ": the field was baked with a different sigma than the config now records");
        }
        return *fromFile;
    }
    if (fromFile) return *fromFile;
    if (fromJson) return *fromJson;
    throw std::runtime_error(
        "no sigma in " + path.string() + " (no \"sigma\" grid) and no usable sidecar; "
        "pass an explicit sigma or re-bake with the current generator");
}

} // namespace mf

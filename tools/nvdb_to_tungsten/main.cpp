#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Options {
    fs::path config;
    fs::path output;
    fs::path vdb;
    int haloVoxels = 3;
};

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ConvertedField {
    double voxelSize = 0.0;
    double sigma = 0.0;
    std::uint64_t sourceActiveVoxelCount = 0;
    std::uint64_t haloVoxelCount = 0;
    int haloVoxels = 0;
    Vec3d minimum;
    Vec3d maximum;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void printUsage(std::ostream& stream) {
    stream
        << "Usage:\n"
        << "  macrofacet_nvdb_to_tungsten --config <macrofacet.json>\n"
        << "      --output <tungsten.json> [--vdb <sdf.vdb>]\n"
        << "      [--halo-voxels <n>]\n\n"
        << "The input must use field.mean_type=\"nanovdb\". The converter copies\n"
        << "the NanoVDB grid named \"sdf\" to an OpenVDB FloatGrid, adds a\n"
        << "positive SDF halo (default: 3 voxels), and emits a\n"
        << "sparse-conv-gpis-tungsten scene that reads it as a tabulated mean.\n";
}

int parseHaloVoxels(const std::string& text) {
    std::size_t parsed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &parsed);
    } catch (const std::exception&) {
        fail("--halo-voxels must be an integer of at least 3");
    }
    if (parsed != text.size() || value < 3 || value > std::numeric_limits<int>::max()) {
        fail("--halo-voxels must be an integer of at least 3");
    }
    return static_cast<int>(value);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            printUsage(std::cout);
            std::exit(0);
        }
        if (argument != "--config" && argument != "--output" && argument != "--vdb" &&
            argument != "--halo-voxels") {
            fail("unknown argument: " + argument);
        }
        if (++i == argc) fail("missing value after " + argument);
        if (argument == "--config") options.config = argv[i];
        else if (argument == "--output") options.output = argv[i];
        else if (argument == "--vdb") options.vdb = argv[i];
        else options.haloVoxels = parseHaloVoxels(argv[i]);
    }
    if (options.config.empty() || options.output.empty()) {
        printUsage(std::cerr);
        fail("--config and --output are required");
    }
    if (options.vdb.empty()) {
        options.vdb = options.output.parent_path() /
                      (options.output.stem().string() + "_sdf.vdb");
    }
    return options;
}

fs::path absoluteNormalized(const fs::path& path) {
    std::error_code error;
    fs::path result = fs::absolute(path, error);
    if (error) fail("cannot make path absolute: " + path.string() + ": " + error.message());
    return result.lexically_normal();
}

void createParentDirectory(const fs::path& path) {
    if (path.parent_path().empty()) return;
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    if (error) fail("cannot create directory " + path.parent_path().string() + ": " + error.message());
}

Json readJson(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) fail("cannot open JSON file: " + path.string());
    try {
        Json value;
        stream >> value;
        return value;
    } catch (const std::exception& error) {
        fail("cannot parse JSON file " + path.string() + ": " + error.what());
    }
}

const Json& requireObject(const Json& parent, const char* key) {
    if (!parent.contains(key) || !parent.at(key).is_object()) {
        fail(std::string("missing JSON object: ") + key);
    }
    return parent.at(key);
}

std::string requireString(const Json& parent, const char* key) {
    if (!parent.contains(key) || !parent.at(key).is_string()) {
        fail(std::string("missing JSON string: ") + key);
    }
    return parent.at(key).get<std::string>();
}

double positiveNumber(const Json& parent, const char* key) {
    if (!parent.contains(key) || !parent.at(key).is_number()) {
        fail(std::string("missing JSON number: ") + key);
    }
    const double value = parent.at(key).get<double>();
    if (!(value > 0.0) || !std::isfinite(value)) {
        fail(std::string(key) + " must be finite and positive");
    }
    return value;
}

fs::path resolveInputPath(const fs::path& value, const fs::path& config) {
    if (value.is_absolute()) return value.lexically_normal();
    if (fs::exists(value)) return absoluteNormalized(value);
    const fs::path besideConfig = config.parent_path() / value;
    if (fs::exists(besideConfig)) return absoluteNormalized(besideConfig);
    fail("cannot find input file: " + value.string() +
         " (tried the working directory and the config directory)");
}

fs::path resolveOutputPath(const fs::path& value) {
    if (value.is_absolute()) return value.lexically_normal();
    // Macrofacet currently interprets relative paths from its working directory.
    // Keep that convention for generated output paths as well.
    return absoluteNormalized(value);
}

std::string pathRelativeTo(const fs::path& path, const fs::path& base) {
    std::error_code error;
    fs::path relative = fs::relative(path, base, error);
    if (!error && !relative.empty()) return relative.generic_string();
    return path.generic_string();
}

std::array<double, 3> jsonVec3(const Json& parent, const char* key) {
    if (!parent.contains(key) || !parent.at(key).is_array() || parent.at(key).size() != 3) {
        fail(std::string(key) + " must be an array of three numbers");
    }
    std::array<double, 3> result{};
    for (std::size_t i = 0; i < 3; ++i) {
        if (!parent.at(key).at(i).is_number()) {
            fail(std::string(key) + " must be an array of three numbers");
        }
        result[i] = parent.at(key).at(i).get<double>();
        if (!std::isfinite(result[i])) fail(std::string(key) + " contains a non-finite value");
    }
    return result;
}

Json toJson(const Vec3d& value) {
    return Json::array({value.x, value.y, value.z});
}

Vec3d add(const Vec3d& a, const Vec3d& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3d subtract(const Vec3d& a, const Vec3d& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3d multiply(const Vec3d& value, double scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

double length(const Vec3d& value) {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vec3d nanoPoint(const nanovdb::Vec3d& value) {
    return {value[0], value[1], value[2]};
}

bool close(double a, double b, double scale) {
    return std::abs(a - b) <= 1.0e-9 * std::max(1.0, scale);
}

double validateNanoTransform(const nanovdb::NanoGrid<float>& grid, Vec3d& origin) {
    origin = nanoPoint(grid.indexToWorld(nanovdb::Vec3d(0.0, 0.0, 0.0)));
    const Vec3d ex = subtract(nanoPoint(grid.indexToWorld(nanovdb::Vec3d(1.0, 0.0, 0.0))), origin);
    const Vec3d ey = subtract(nanoPoint(grid.indexToWorld(nanovdb::Vec3d(0.0, 1.0, 0.0))), origin);
    const Vec3d ez = subtract(nanoPoint(grid.indexToWorld(nanovdb::Vec3d(0.0, 0.0, 1.0))), origin);
    const double dx = ex.x;
    if (!(dx > 0.0) || !std::isfinite(dx) ||
        !close(ex.y, 0.0, dx) || !close(ex.z, 0.0, dx) ||
        !close(ey.x, 0.0, dx) || !close(ey.y, dx, dx) || !close(ey.z, 0.0, dx) ||
        !close(ez.x, 0.0, dx) || !close(ez.y, 0.0, dx) || !close(ez.z, dx, dx)) {
        fail("the NanoVDB transform is rotated, reflected, or non-uniform; "
             "sparse-conv-gpis-tungsten's VdbGrid cannot preserve that transform");
    }
    return dx;
}

double readSigma(const fs::path& nvdb, const Json& field, const Json* sidecar) {
    double sigma = 0.0;
    if (nanovdb::io::hasGrid(nvdb.string(), "sigma")) {
        auto handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdb.string(), "sigma");
        const auto* grid = handle.grid<float>();
        if (!grid) fail("the NanoVDB grid named \"sigma\" is not a float grid");
        sigma = static_cast<double>(grid->tree().root().background());
    } else if (field.contains("sigma") && field.at("sigma").is_number()) {
        sigma = field.at("sigma").get<double>();
    } else if (sidecar && sidecar->contains("sigma") && sidecar->at("sigma").is_number()) {
        sigma = sidecar->at("sigma").get<double>();
    }
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        fail("cannot obtain a finite positive sigma from the NanoVDB, field config, or sidecar");
    }
    return sigma;
}

ConvertedField convertSdf(const fs::path& nvdb, const fs::path& vdb,
                          const Json& field, const Json* sidecar, int haloVoxels) {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    try {
        handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdb.string(), "sdf");
    } catch (const std::exception& error) {
        fail("cannot read float grid \"sdf\" from " + nvdb.string() + ": " + error.what());
    }
    const auto* source = handle.grid<float>();
    if (!source) fail("the NanoVDB grid named \"sdf\" is not a float grid");

    Vec3d origin;
    const double dx = validateNanoTransform(*source, origin);
    const nanovdb::CoordBBox bbox = source->indexBBox();
    const auto minCoord = bbox.min();
    const auto maxCoord = bbox.max();
    if (minCoord[0] > maxCoord[0] || minCoord[1] > maxCoord[1] || minCoord[2] > maxCoord[2]) {
        fail("the NanoVDB sdf grid has an empty active bounding box");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (minCoord[axis] < std::numeric_limits<int>::min() + haloVoxels ||
            maxCoord[axis] > std::numeric_limits<int>::max() - haloVoxels) {
            fail("the NanoVDB sdf index bounds cannot be expanded by the requested halo");
        }
    }

    openvdb::initialize();
    auto destination = openvdb::FloatGrid::create(source->tree().root().background());
    destination->setName("sdf");
    destination->setGridClass(openvdb::GRID_UNKNOWN);
    auto transform = openvdb::math::Transform::createLinearTransform(dx);
    transform->postTranslate(openvdb::Vec3d(origin.x, origin.y, origin.z));
    const openvdb::Vec3d checkOrigin = transform->indexToWorld(openvdb::Vec3d(0.0));
    if (!close(checkOrigin.x(), origin.x, dx) || !close(checkOrigin.y(), origin.y, dx) ||
        !close(checkOrigin.z(), origin.z, dx)) {
        fail("internal error while constructing the OpenVDB transform");
    }
    destination->setTransform(transform);

    auto sourceAccessor = source->getAccessor();
    auto destinationAccessor = destination->getAccessor();
    std::uint64_t sourceActiveCount = 0;
    for (int z = minCoord[2]; z <= maxCoord[2]; ++z) {
        for (int y = minCoord[1]; y <= maxCoord[1]; ++y) {
            for (int x = minCoord[0]; x <= maxCoord[0]; ++x) {
                const nanovdb::Coord coordinate(x, y, z);
                if (!sourceAccessor.isActive(coordinate)) continue;
                const float value = sourceAccessor.getValue(coordinate);
                if (!std::isfinite(value)) fail("the NanoVDB sdf grid contains a non-finite value");
                destinationAccessor.setValueOn(openvdb::Coord(x, y, z), value);
                ++sourceActiveCount;
            }
        }
    }
    if (sourceActiveCount == 0) fail("the NanoVDB sdf grid has no active voxels");

    // Tungsten clamps emission-less VDB queries to [activeMin + 2, activeMax - 2]
    // (its upper bound is exclusive before subtracting three).  Grow the active
    // bbox without changing the process box so every point in the original domain
    // remains outside that clamp zone.  Continue the SDF outward from the nearest
    // original boundary node; an invalid non-positive boundary is rejected instead
    // of manufacturing a second zero surface in the halo.
    const int expandedMin[3] = {minCoord[0] - haloVoxels, minCoord[1] - haloVoxels,
                                minCoord[2] - haloVoxels};
    const int expandedMax[3] = {maxCoord[0] + haloVoxels, maxCoord[1] + haloVoxels,
                                maxCoord[2] + haloVoxels};
    std::uint64_t haloCount = 0;
    for (int z = expandedMin[2]; z <= expandedMax[2]; ++z) {
        for (int y = expandedMin[1]; y <= expandedMax[1]; ++y) {
            for (int x = expandedMin[0]; x <= expandedMax[0]; ++x) {
                if (x >= minCoord[0] && x <= maxCoord[0] &&
                    y >= minCoord[1] && y <= maxCoord[1] &&
                    z >= minCoord[2] && z <= maxCoord[2]) {
                    continue;
                }

                const int nearestX = std::clamp(x, minCoord[0], maxCoord[0]);
                const int nearestY = std::clamp(y, minCoord[1], maxCoord[1]);
                const int nearestZ = std::clamp(z, minCoord[2], maxCoord[2]);
                const nanovdb::Coord nearest(nearestX, nearestY, nearestZ);
                if (!sourceAccessor.isActive(nearest)) {
                    fail("cannot construct a positive SDF halo: the original full-domain "
                         "grid has an inactive boundary voxel");
                }
                const float boundaryValue = sourceAccessor.getValue(nearest);
                if (!(boundaryValue > 0.0f) || !std::isfinite(boundaryValue)) {
                    fail("cannot construct a positive SDF halo: the original full-domain "
                         "grid has a non-positive or non-finite boundary value");
                }

                const double deltaX = static_cast<double>(x - nearestX);
                const double deltaY = static_cast<double>(y - nearestY);
                const double deltaZ = static_cast<double>(z - nearestZ);
                const float value = static_cast<float>(
                    static_cast<double>(boundaryValue) +
                    dx * std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ));
                if (!(value > 0.0f) || !std::isfinite(value)) {
                    fail("the generated SDF halo contains a non-positive or non-finite value");
                }
                destinationAccessor.setValueOn(openvdb::Coord(x, y, z), value);
                ++haloCount;
            }
        }
    }

    const openvdb::CoordBBox outputBounds = destination->evalActiveVoxelBoundingBox();
    const openvdb::Coord expectedMin(expandedMin[0], expandedMin[1], expandedMin[2]);
    const openvdb::Coord expectedMax(expandedMax[0], expandedMax[1], expandedMax[2]);
    if (outputBounds.min() != expectedMin || outputBounds.max() != expectedMax) {
        fail("internal error: the generated OpenVDB halo has unexpected active bounds");
    }
    // Reproduce Tungsten's clamp limits. Its VdbGrid bounds.max() is activeMax + 1.
    const openvdb::Coord clampMin = outputBounds.min().offsetBy(2);
    const openvdb::Coord clampMax = outputBounds.max().offsetBy(-2);
    for (int axis = 0; axis < 3; ++axis) {
        if (minCoord[axis] < clampMin[axis] || maxCoord[axis] > clampMax[axis]) {
            fail("internal error: the generated halo does not protect the original domain "
                 "from Tungsten's boundary clamp");
        }
    }

    createParentDirectory(vdb);
    openvdb::GridPtrVec grids;
    grids.push_back(destination);
    openvdb::io::File file(vdb.string());
    file.write(grids);
    file.close();

    const Vec3d minimum = nanoPoint(source->indexToWorld(nanovdb::Vec3d(
        static_cast<double>(minCoord[0]), static_cast<double>(minCoord[1]),
        static_cast<double>(minCoord[2]))));
    const Vec3d maximum = nanoPoint(source->indexToWorld(nanovdb::Vec3d(
        static_cast<double>(maxCoord[0]), static_cast<double>(maxCoord[1]),
        static_cast<double>(maxCoord[2]))));
    return {dx, readSigma(nvdb, field, sidecar), sourceActiveCount, haloCount,
            haloVoxels, minimum, maximum};
}

double correlationLength(const Json& root, const Json& field, const Json* sidecar,
                         double sigma) {
    if (field.contains("correlation_lengths")) {
        const auto lengths = jsonVec3(field, "correlation_lengths");
        for (double value : lengths) {
            if (!(value > 0.0)) fail("field.correlation_lengths must be positive");
        }
        const double tolerance = 1.0e-9 * std::max({1.0, lengths[0], lengths[1], lengths[2]});
        if (std::abs(lengths[0] - lengths[1]) > tolerance ||
            std::abs(lengths[0] - lengths[2]) > tolerance) {
            fail("anisotropic field.correlation_lengths are not converted by this preset");
        }
        return lengths[0];
    }

    const Json& material = requireObject(root, "material");
    double roughness = 0.0;
    if (material.contains("roughness") && material.at("roughness").is_number()) {
        roughness = material.at("roughness").get<double>();
    } else if (sidecar && sidecar->contains("alpha") && sidecar->at("alpha").is_number()) {
        roughness = sidecar->at("alpha").get<double>();
    }
    if (!(roughness > 0.0) || !std::isfinite(roughness)) {
        fail("cannot derive the covariance length: provide material.roughness or "
             "isotropic field.correlation_lengths");
    }
    return std::sqrt(2.0) * sigma / roughness;
}

Json makeScene(const Json& source, const ConvertedField& converted,
               const fs::path& vdb, const fs::path& outputJson, const Json* sidecar) {
    const Json& field = requireObject(source, "field");
    const Json& material = requireObject(source, "material");
    const Json& render = requireObject(source, "render");

    if (material.value("type", std::string()) != "conductor") {
        fail("only material.type=\"conductor\" is currently supported");
    }
    const auto eta = jsonVec3(material, "eta_rgb");
    const auto k = jsonVec3(material, "k_rgb");
    const double ell = correlationLength(source, field, sidecar, converted.sigma);

    const int width = render.value("width", 512);
    const int height = render.value("height", 512);
    const int spp = render.value("samples_per_pixel", 1);
    const double fov = positiveNumber(render, "vertical_fov_degrees");
    if (width <= 0 || height <= 0 || spp <= 0) fail("render dimensions and samples_per_pixel must be positive");
    const auto cameraPosition = jsonVec3(render, "camera_position");
    const auto cameraTarget = jsonVec3(render, "camera_target");
    Vec3d view{cameraTarget[0] - cameraPosition[0], cameraTarget[1] - cameraPosition[1],
               cameraTarget[2] - cameraPosition[2]};
    if (!(length(view) > 0.0)) fail("camera_position and camera_target must differ");
    const Json up = std::abs(view.z / length(view)) > 0.999
        ? Json::array({0.0, 1.0, 0.0}) : Json::array({0.0, 0.0, 1.0});

    const Vec3d extent = subtract(converted.maximum, converted.minimum);
    if (!(extent.x > 0.0 && extent.y > 0.0 && extent.z > 0.0)) {
        fail("the sdf active-node domain is degenerate");
    }
    const Vec3d center = multiply(add(converted.minimum, converted.maximum), 0.5);

    fs::path renderDirectory = "outputs/tungsten";
    if (source.contains("output_directory") && source.at("output_directory").is_string()) {
        renderDirectory = source.at("output_directory").get<std::string>();
        renderDirectory += "_tungsten";
    }
    renderDirectory = resolveOutputPath(renderDirectory);
    const std::string sceneStem = outputJson.stem().string();

    Json grid = {
        {"type", "vdb"},
        {"file", pathRelativeTo(vdb, outputJson.parent_path())},
        {"density_name", "sdf"},
        {"emission_name", ""},
        {"request_sdf", false},
        {"normalize_size", false},
        {"density_scale", 1.0},
        {"integration_method", "raymarching"},
        {"interpolate", "linear"},
        {"transform", {{"position", {0.0, 0.0, 0.0}},
                       {"scale", 1.0}, {"rotation", {0.0, 0.0, 0.0}}}}
    };

    Json medium = {
        {"name", "gp"}, {"type", "sparse_conv_noise"},
        {"phase_function", {{"type", "brdf"},
            {"bsdf", {{"type", "conductor"}, {"albedo", 1.0},
                      {"eta", eta}, {"k", k}}}}},
        {"max_bounces", 64}, {"sigma_a", 0.0}, {"sigma_s", 1.0}, {"density", 1.0},
        {"step_size", 0.5 * converted.voxelSize},
        {"seed", source.value("seed", std::uint64_t{0})},
        {"impulse_density", 3.0}, {"single_realization", false},
        {"1D_sampling", true}, {"1D_sampling_scheme", "uni"},
        {"1D_gradient_correlationXY", false}, {"isotropic_3D_sampling", false},
        {"correlation_context", "renewal+"},
        {"gaussian_process", {{"type", "standard"},
            {"mean", {{"type", "tabulated"}, {"grid", grid}}},
            {"covariance", {{"type", "squared_exponential"},
                            {"lengthScale", ell}, {"sigma", converted.sigma}}}}}
    };

    Json result = {
        {"media", Json::array({medium})},
        {"bsdfs", Json::array({{{"name", "processBox"}, {"type", "forward"}, {"albedo", 1.0}}})},
        {"primitives", Json::array({
            {{"name", "processBox"}, {"type", "cube"}, {"bsdf", "processBox"},
             {"int_medium", "gp"},
             {"transform", {{"position", toJson(center)}, {"scale", toJson(extent)}}}},
            {{"type", "infinite_sphere"}, {"emission", 1.0}, {"scale", 1.0},
             {"sample", true}, {"bsdf", {{"type", "null"}, {"albedo", 1.0}}}}
        })},
        {"camera", {{"type", "pinhole"}, {"tonemap", "linear"},
                    {"resolution", {width, height}}, {"reconstruction_filter", "tent"},
                    {"transform", {{"position", cameraPosition}, {"look_at", cameraTarget},
                                   {"up", up}, {"scale", {-1.0, 1.0, 1.0}}}},
                    {"fov", fov}}},
        {"integrator", {{"type", "path_tracer"}, {"min_bounces", 0}, {"max_bounces", 20},
                        {"enable_light_sampling", true}, {"enable_volume_light_sampling", true}}},
        {"renderer", {{"output_directory", pathRelativeTo(renderDirectory, outputJson.parent_path())},
                      {"overwrite_output_files", true}, {"adaptive_sampling", false},
                      {"enable_resume_render", false}, {"stratified_sampler", false},
                      {"scene_bvh", true}, {"spp", spp}, {"spp_step", 16},
                      {"output_file", sceneStem + ".png"},
                      {"hdr_output_file", sceneStem + ".pfm"}}}
    };
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options rawOptions = parseOptions(argc, argv);
        const fs::path configPath = absoluteNormalized(rawOptions.config);
        const fs::path outputPath = absoluteNormalized(rawOptions.output);
        const fs::path vdbPath = absoluteNormalized(rawOptions.vdb);
        if (!fs::exists(configPath)) fail("config file does not exist: " + configPath.string());

        const Json source = readJson(configPath);
        const Json& field = requireObject(source, "field");
        if (field.value("mean_type", std::string()) != "nanovdb") {
            fail("this converter only accepts field.mean_type=\"nanovdb\"");
        }
        const fs::path nvdbPath = resolveInputPath(requireString(field, "grid_file"), configPath);

        Json sidecarStorage;
        const Json* sidecar = nullptr;
        const fs::path sidecarPath = fs::path(nvdbPath.string() + ".json");
        if (fs::exists(sidecarPath)) {
            sidecarStorage = readJson(sidecarPath);
            sidecar = &sidecarStorage;
            if (sidecarStorage.value("version", 0) != 2) {
                fail("the NanoVDB sidecar is not version 2; re-bake it with the current "
                     "index-node convention before converting");
            }
            if (sidecarStorage.value("coverage", std::string()) != "full_domain") {
                fail("the NanoVDB sidecar does not declare coverage=\"full_domain\"; "
                     "a surface-band SDF cannot be directly converted safely");
            }
        } else {
            std::cerr << "warning: no .nvdb.json sidecar was found; full-domain coverage cannot be verified\n";
        }

        const ConvertedField converted = convertSdf(
            nvdbPath, vdbPath, field, sidecar, rawOptions.haloVoxels);
        const Json tungsten = makeScene(source, converted, vdbPath, outputPath, sidecar);
        createParentDirectory(outputPath);
        std::ofstream stream(outputPath);
        if (!stream) fail("cannot write output JSON: " + outputPath.string());
        stream << std::setw(4) << tungsten << '\n';
        if (!stream) fail("failed while writing output JSON: " + outputPath.string());

        const std::string environment = requireObject(source, "render").value("environment", "unit_white");
        if (environment != "unit_white") {
            std::cerr << "warning: macrofacet environment \"" << environment
                      << "\" was replaced by a constant-white infinite sphere\n";
        }
        std::cout << "Wrote OpenVDB SDF: " << vdbPath.string() << '\n'
                  << "Wrote Tungsten scene: " << outputPath.string() << '\n'
                  << "Source active voxels: " << converted.sourceActiveVoxelCount << '\n'
                  << "Positive halo voxels: " << converted.haloVoxelCount << '\n'
                  << "Halo width: " << converted.haloVoxels << " voxels\n"
                  << "Voxel size: " << std::setprecision(17) << converted.voxelSize << '\n'
                  << "Sigma: " << converted.sigma << '\n'
                  << "The process box keeps the original active-node domain; Tungsten's "
                     "existing boundary clamp is unchanged.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

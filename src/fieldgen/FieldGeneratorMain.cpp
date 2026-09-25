// macrofacet_fieldgen -- precompute a NanoVDB model field.
//
// This is the reimplementation of the original pbrt-based
// macrofcaet_vdb_generator: read a mesh, build a signed distance field with
// libigl, and write the (density, alpha, sdf, sigma) grids the renderer samples.
// See docs/archive/PLAN_NANOVDB_FIELD.md 2 for the original specification.

#include "fieldgen/NanoVdbIO.h"
#include "fieldgen/PlyReader.h"
#include "fieldgen/SyntheticMesh.h"
#include "fieldgen/VdbBaker.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string meshPath;
    std::string outputPath = "out.nvdb";
    mf::BakeSettings bake;
    bool allowSparse = false;
    // Synthetic mesh, so the pipeline can be exercised (and checked against an
    // analytic sphere) without shipping a .ply asset.
    double sphereRadius = 0.0;
    int sphereSegments = 128;
    // Imported meshes arrive in arbitrary units, but sigma is a world-space
    // length and the band is +-bandSigmas of it. Without this a model authored
    // in, say, centimetres gets a band that swallows the whole object.
    double fitExtent = 0.0;  // <= 0 leaves the size alone
    bool center = false;
};

void printUsage() {
    std::printf(
        "usage: macrofacet_fieldgen <mesh.ply> [options]\n"
        "\n"
        "  --out <file.nvdb>     output path (default out.nvdb; a .json sidecar is written too)\n"
        "  --sigma <s>           surface position standard deviation (default 0.05)\n"
        "  --alpha <a>           material NDF parameter written to the alpha grid (default 0.5)\n"
        "  --x --y --z <n>       voxels per axis over the mesh bounding box (default 32)\n"
        "  --band <k>            write the grids only inside +-k sigma (default 3)\n"
        "  --background <k>      sdf background, in sigma (default 6)\n"
        "  --threads <n>         bake threads (default: hardware concurrency)\n"
        "  --symmetric           use |distance| in Density() instead of the signed value\n"
        "  --sign-mode <m>       winding (default) | threshold\n"
        "  --no-prune-band       query the winding number for every voxel (slower, and\n"
        "                        the only correct choice for a non-watertight mesh)\n"
        "  --full-domain         retain signed distance throughout the baked box\n"
        "  --allow-sparse        do not fail when dx is too coarse for the band\n"
        "\n"
        "  --fit <extent>        scale the mesh so its largest extent is <extent> world\n"
        "                        units (sigma is a world-space length, so an imported\n"
        "                        model has to be brought to the scale sigma implies)\n"
        "  --center              move the mesh's bounding-box centre to the origin\n"
        "                        (applied before --fit; both are recorded in the sidecar)\n"
        "\n"
        "  --sphere <r>          bake a synthetic lat/long sphere of radius r instead of a file\n"
        "  --sphere-segments <n> longitude segments for --sphere (default 128)\n");
}

std::string requireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            std::exit(0);
        } else if (argument == "--out") {
            options.outputPath = requireValue(argc, argv, index);
        } else if (argument == "--sigma") {
            options.bake.sigma = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--alpha") {
            options.bake.alpha = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--x") {
            options.bake.resolution[0] = std::stoi(requireValue(argc, argv, index));
        } else if (argument == "--y") {
            options.bake.resolution[1] = std::stoi(requireValue(argc, argv, index));
        } else if (argument == "--z") {
            options.bake.resolution[2] = std::stoi(requireValue(argc, argv, index));
        } else if (argument == "--band") {
            options.bake.bandSigmas = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--background") {
            options.bake.backgroundSigmas = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--threads") {
            options.bake.threadCount = std::stoi(requireValue(argc, argv, index));
        } else if (argument == "--symmetric") {
            options.bake.symmetricDensity = true;
        } else if (argument == "--sign-mode") {
            const std::string mode = requireValue(argc, argv, index);
            if (mode == "winding") options.bake.signMode = mf::SignMode::Winding;
            else if (mode == "threshold") options.bake.signMode = mf::SignMode::Threshold;
            else throw std::invalid_argument("unknown sign mode: " + mode);
        } else if (argument == "--no-prune-band") {
            options.bake.pruneByUnsignedDistance = false;
        } else if (argument == "--full-domain") {
            options.bake.fullDomain = true;
        } else if (argument == "--allow-sparse") {
            options.allowSparse = true;
        } else if (argument == "--fit") {
            options.fitExtent = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--center") {
            options.center = true;
        } else if (argument == "--sphere") {
            options.sphereRadius = std::stod(requireValue(argc, argv, index));
        } else if (argument == "--sphere-segments") {
            options.sphereSegments = std::stoi(requireValue(argc, argv, index));
        } else if (!argument.empty() && argument[0] == '-') {
            throw std::invalid_argument("unknown option: " + argument);
        } else if (options.meshPath.empty()) {
            options.meshPath = argument;
        } else {
            throw std::invalid_argument("unexpected extra argument: " + argument);
        }
    }
    if (options.meshPath.empty() && !(options.sphereRadius > 0.0)) {
        throw std::invalid_argument("no input mesh: pass a .ply path or --sphere <radius>");
    }
    return options;
}

double fileSizeMegabytes(const std::filesystem::path& path) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) return 0.0;
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);

        double coarseDx = 0.0;
        mf::TriangleMesh mesh;
        if (options.sphereRadius > 0.0) {
            mesh = mf::makeSphereMesh(options.sphereRadius, options.sphereSegments);
            std::printf("mesh : synthetic sphere r=%.4g (%ld vertices, %ld triangles)\n",
                        options.sphereRadius, static_cast<long>(mesh.vertices.rows()),
                        static_cast<long>(mesh.faces.rows()));
        } else {
            mesh = mf::readPlyMesh(options.meshPath);
            std::printf("mesh : %s (%ld vertices, %ld triangles)\n", options.meshPath.c_str(),
                        static_cast<long>(mesh.vertices.rows()),
                        static_cast<long>(mesh.faces.rows()));
        }

        // Before bandIsResolvable: dx, the origin and every bound the bake
        // reports are derived from the mesh, so the transform has to land first.
        const mf::MeshTransform transform =
            mf::fitMeshTransform(mesh, options.fitExtent, options.center);
        mf::applyMeshTransform(mesh, transform);
        if (!transform.isIdentity()) {
            const mf::Bounds3 fitted = mesh.bounds();
            const mf::Vector3 extent = fitted.maximum - fitted.minimum;
            std::printf("xform: scale=%.6g translation=(%.4g,%.4g,%.4g) -> extent (%.4g,%.4g,%.4g)\n",
                        transform.scale, transform.translation.x(), transform.translation.y(),
                        transform.translation.z(), extent.x(), extent.y(), extent.z());
        }

        // dx must stay below sigma or the band holds no voxels at all -- which
        // otherwise shows up much later as a grid that reads as uniform
        // background everywhere.
        if (!mf::bandIsResolvable(mesh, options.bake, coarseDx)) {
            const mf::Bounds3 bounds = mesh.bounds();
            const mf::Vector3 gap = bounds.maximum - bounds.minimum;
            const double recommended = std::ceil(gap.maxCoeff() / (options.bake.sigma / 2.0));
            std::fprintf(stderr,
                         "warning: dx = %.6g exceeds sigma = %.6g, so the +-%.3g sigma band "
                         "may contain no voxels.\nsuggested: roughly %.0f voxels per axis "
                         "(try --x %.0f --y %.0f --z %.0f)\n",
                         coarseDx, options.bake.sigma, options.bake.bandSigmas, recommended,
                         recommended, recommended, recommended);
            if (!options.allowSparse) {
                std::fprintf(stderr, "refusing to bake; pass --allow-sparse to proceed anyway\n");
                return 2;
            }
        }

        // The bake is the expensive part; failing on a missing output directory
        // after it finished would be a poor trade. Same behaviour as
        // macrofacet_experiments, which creates its --output directory.
        const std::filesystem::path outputPath(options.outputPath);
        if (outputPath.has_parent_path()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        const auto start = std::chrono::steady_clock::now();
        mf::BakedGrids baked = mf::bakeMacrofacetField(mesh, options.bake);
        const auto baked_ = std::chrono::steady_clock::now();

        mf::writeFieldFile(options.outputPath, baked, options.bake, options.meshPath, transform);
        const auto written = std::chrono::steady_clock::now();

        const mf::BakeReport& report = baked.report;
        const double bakeSeconds = std::chrono::duration<double>(baked_ - start).count();
        const double writeSeconds = std::chrono::duration<double>(written - baked_).count();
        std::printf("grid : dx=%.6g origin=(%.4g,%.4g,%.4g) voxels=%lldx%lldx%lld threads=%d\n",
                    report.dx, report.origin.x(), report.origin.y(), report.origin.z(),
                    report.voxels[0], report.voxels[1], report.voxels[2], report.threads);
        std::printf("stored: %lld voxels %s (%.4g%% of the grid)\n", report.bandVoxels,
                    options.bake.fullDomain ? "in the full box" : "in the surface band",
                    100.0 * static_cast<double>(report.bandVoxels) /
                        (static_cast<double>(report.voxels[0]) * report.voxels[1] * report.voxels[2]));
        std::printf("sdf  : range [%.6g, %.6g]  background = %.6g (= %.3g sigma)\n",
                    report.sdfMinimum, report.sdfMaximum, report.background,
                    report.background / options.bake.sigma);
        std::printf("grad : maximum |grad m| bound = %.6g  (density max = %.6g)\n",
                    report.maximumGradientNorm, report.densityMaximum);
        std::printf("write: %s (%.2f MB) + %s.json in %.2f s; bake took %.2f s\n",
                    options.outputPath.c_str(), fileSizeMegabytes(options.outputPath),
                    options.outputPath.c_str(), writeSeconds, bakeSeconds);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

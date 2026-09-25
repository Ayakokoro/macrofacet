#pragma once

#include "fieldgen/VdbBaker.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mf {

// The JSON written next to a .nvdb. It is a convenience, never the source of
// truth: everything the renderer needs is also inside the .nvdb itself (the
// sigma grid carries sigma, every grid carries its own background).
struct FieldSidecar {
    int version = 1;
    double sigma = 0.0;
    double alpha = 0.0;
    double dx = 0.0;
    Point3 origin = Point3::Zero();
    double band = 0.0;
    bool fullDomain = false;
    double sdfBackground = 0.0;
    std::string signMode = "winding";
    std::vector<std::string> grids;
    std::array<int, 3> resolution{0, 0, 0};
    std::array<long long, 3> voxels{0, 0, 0};
    std::string meshName;
    double maximumGradientNorm = 0.0;
    double sdfMinimum = 0.0;
    double sdfMaximum = 0.0;
};

// Writes the four grids to `path` and the sidecar to `<path>.json`.
//
// `transform` is recorded for provenance only -- it is deliberately not read
// back into FieldSidecar, because nothing downstream consumes it (the renderer
// needs sigma/dx/origin/background, all of which live inside the .nvdb). It is
// written because it is the one input to a bake that the .nvdb cannot recover.
void writeFieldFile(const std::filesystem::path& path, const BakedGrids& grids,
                    const BakeSettings& settings, const std::string& meshName,
                    const MeshTransform& transform = MeshTransform{});

// Reads the constant sigma grid. std::nullopt when the file simply has no grid
// by that name; a missing or unreadable file still throws.
std::optional<double> readSigmaGrid(const std::filesystem::path& path);

// std::nullopt when there is no sidecar. Throws on a malformed one.
std::optional<FieldSidecar> readSidecar(const std::filesystem::path& path);

// The rule from PLAN 2.4: prefer the sigma stored inside the .nvdb, fall back
// to the sidecar. When both exist and disagree this throws rather than picking
// one -- that combination means sigma changed without a re-bake, and silently
// choosing would render at the wrong statistical scale with nothing to show
// for it.
double resolveSigma(const std::filesystem::path& path);

// Whether two sigmas are the same sigma, i.e. whether they agree to the
// precision a .nvdb can hold. The sigma grid stores a float32, so a value that
// round-tripped through a file differs from the double that was written by up
// to one float32 ulp -- and at most sigmas that ulp is *wider* than the
// absolute 1e-9 tolerance this used to test against, which rejected 0.06, 0.08,
// 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, ... That made the generator's own output
// unreadable at those sigmas and a config that restated its sigma a hard error.
// A genuinely different sigma is nowhere near an ulp away, so nothing is lost.
bool sameSigma(double a, double b);

} // namespace mf

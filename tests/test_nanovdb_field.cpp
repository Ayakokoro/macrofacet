#include "TestHarness.h"

#include "fieldgen/NanoVdbIO.h"
#include "fieldgen/SyntheticMesh.h"
#include "fieldgen/VdbBaker.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/NanoVdbSampledField.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/macrofacet/ClassicCoefficients.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/NarrowBandMedium.h"

#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;

// A lat/long sphere is inscribed in the sphere it approximates, so its surface
// sits up to the sagitta inside radius r. With nu = 128 the dominant term is the
// pole-to-pole spacing: 1 - cos(pi/(nu/2)) ~ 1.2e-3.
double facetedSphereSagitta(double radius, int segments) {
    const int nv = std::max(4, std::max(8, segments) / 2);
    return radius * (1.0 - std::cos(kPi / nv));
}

std::filesystem::path tempFieldPath(const std::string& name) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "macrofacet_field_test";
    std::filesystem::create_directories(directory);
    return directory / name;
}

// Bakes a sphere once and returns the file it wrote; every case below reads that
// same file back, which is also what the renderer does.
struct BakedSphere {
    std::filesystem::path path;
    mf::BakeReport report;
    mf::BakeSettings settings;
    double radius = 1.0;
    int segments = 128;
};

const BakedSphere& sphereField() {
    static const BakedSphere baked = [] {
        BakedSphere result;
        result.radius = 1.0;
        result.segments = 128;
        result.settings.sigma = 0.05;
        result.settings.alpha = 0.5;
        result.settings.resolution = {64, 64, 64};
        result.settings.bandSigmas = 3.0;
        result.settings.backgroundSigmas = 6.0;
        result.path = tempFieldPath("sphere_r1_s64.nvdb");

        const mf::TriangleMesh mesh =
            mf::makeSphereMesh(result.radius, result.segments);
        mf::BakedGrids grids = mf::bakeMacrofacetField(mesh, result.settings);
        result.report = grids.report;
        mf::writeFieldFile(result.path, grids, result.settings, "<synthetic sphere>");
        return result;
    }();
    return baked;
}

// The generator's contract, checked on the raw file rather than through the
// sampler. Every voxel in the grid must read back as exactly one of two things:
//
//   in the band   -> density > 0, alpha = the configured alpha
//   outside       -> sdf = the +6 sigma background, density = 0, alpha = alpha
//
// Anything else is a voxel the renderer would misinterpret. Together with the
// active-voxel counts below this also pins down the active masks: density can
// only be non-zero where its own mask is set, so a mask that disagreed with the
// sdf grid's would show up as a non-zero density over a background sdf (or as a
// zero density inside the band).
void testVoxelContract(TestContext& context) {
    const BakedSphere& baked = sphereField();
    const double sigma = baked.settings.sigma;
    const double band = baked.settings.bandSigmas * sigma;
    const float background = static_cast<float>(baked.settings.backgroundSigmas * sigma);
    const float alphaValue = static_cast<float>(baked.settings.alpha);

    auto density = nanovdb::io::readGrid<nanovdb::HostBuffer>(baked.path.string(), "density");
    auto alpha = nanovdb::io::readGrid<nanovdb::HostBuffer>(baked.path.string(), "alpha");
    auto sdf = nanovdb::io::readGrid<nanovdb::HostBuffer>(baked.path.string(), "sdf");
    auto* densityGrid = density.grid<float>();
    auto* alphaGrid = alpha.grid<float>();
    auto* sdfGrid = sdf.grid<float>();

    const std::uint64_t densityActive = densityGrid->activeVoxelCount();
    context.require(densityActive > 0, "baked density grid has active voxels");
    context.require(densityActive == static_cast<std::uint64_t>(baked.report.bandVoxels),
                    "the active density voxel count equals the reported band size");
    context.require(alphaGrid->activeVoxelCount() == densityActive,
                    "alpha is active on as many voxels as density (" +
                        std::to_string(alphaGrid->activeVoxelCount()) + " vs " +
                        std::to_string(densityActive) + ")");
    context.require(sdfGrid->activeVoxelCount() == densityActive,
                    "sdf is active on as many voxels as density (" +
                        std::to_string(sdfGrid->activeVoxelCount()) + " vs " +
                        std::to_string(densityActive) + ")");

    auto densityAccessor = densityGrid->getAccessor();
    auto alphaAccessor = alphaGrid->getAccessor();
    auto sdfAccessor = sdfGrid->getAccessor();

    const nanovdb::CoordBBox bbox = sdfGrid->indexBBox();
    long long inBand = 0;
    long long outside = 0;
    long long wrong = 0;
    for (int k = bbox.min()[2]; k <= bbox.max()[2]; ++k) {
        for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j) {
            for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i) {
                const nanovdb::Coord ijk(i, j, k);
                const float distance = sdfAccessor.getValue(ijk);
                const float densityValue = densityAccessor.getValue(ijk);
                const float alphaSample = alphaAccessor.getValue(ijk);
                const bool alphaOk = std::abs(alphaSample - alphaValue) <= 1e-6f;

                if (std::abs(distance) < band) {
                    ++inBand;
                    if (!(densityValue > 0.0f) || !alphaOk) ++wrong;
                } else if (distance == background) {
                    ++outside;
                    if (densityValue != 0.0f || !alphaOk) ++wrong;
                } else {
                    // Neither inside the band nor the documented background.
                    ++wrong;
                }
            }
        }
    }

    context.require(inBand > 0, "the sweep found voxels inside the band");
    context.require(outside > 0, "the sweep found voxels reading the background");
    context.require(inBand == static_cast<long long>(densityActive),
                    "every in-band voxel is active (" + std::to_string(inBand) + " vs " +
                        std::to_string(densityActive) + ")");
    context.require(wrong == 0,
                    std::to_string(wrong) + " voxels violate the band/background contract");
}

// Outside the band the file must read back as the documented constants, since
// that is what a ray in vacuum sees.
void testBackgroundValues(TestContext& context) {
    const BakedSphere& baked = sphereField();
    auto mean = mf::NanoVdbMean::open(baked.path);
    auto alphaField = mf::NanoVdbSampledField::open(baked.path, "alpha");

    // sigma travels in a float grid, so it round-trips to float precision
    // (~1e-8 relative), not to double precision.
    const double sigma = baked.settings.sigma;
    context.near(mean->sigma(), sigma, 1e-6 * sigma, "sigma round-trips through the .nvdb");
    context.near(mean->background(), 6.0 * sigma, 1e-6, "sdf background is +6 sigma");
    context.near(alphaField->background(), baked.settings.alpha, 1e-6,
                 "alpha background is the configured alpha");

    // Far outside the object, and inside the object's core -- the bake only
    // covers +-3 sigma, so both are inactive and read as background.
    const double far = 2.0;
    context.near(mean->evaluate(mf::Point3(far, 0.0, 0.0)).value, 6.0 * sigma, 1e-6,
                 "outside the object reads the sdf background");
    context.near(mean->evaluate(mf::Point3(0.0, 0.0, 0.0)).value, 6.0 * sigma, 1e-6,
                 "the object's core reads the sdf background");
    context.near(alphaField->sample(mf::Point3(far, 0.0, 0.0)), baked.settings.alpha, 1e-6,
                 "outside the object reads the alpha background");

    // The gradient must be zero where the field is constant, or the majorant
    // and the collision-gradient sampler both get a phantom slope.
    const mf::MeanJet jet = mean->evaluate(mf::Point3(far, 0.0, 0.0));
    context.near(jet.gradient.norm(), 0.0, 1e-12, "far-field gradient is exactly zero");
}

// The baked sdf has to reproduce the analytic distance to the sphere it was
// built from, to within the two approximations in the way: the faceting of the
// input mesh, and the trilinear interpolation of the grid.
void testSignedDistanceAgreement(TestContext& context) {
    const BakedSphere& baked = sphereField();
    auto mean = mf::NanoVdbMean::open(baked.path);

    const double dx = baked.report.dx;
    const double sagitta = facetedSphereSagitta(baked.radius, baked.segments);
    // Trilinear interpolation of a smooth function on a grid of spacing dx
    // carries O(dx^2 * curvature) error; near a unit sphere that is ~dx^2.
    const double tolerance = sagitta + dx * dx;

    double worst = 0.0;
    int sampled = 0;
    // Sample the band along a few directions, including the poles, where the
    // mesh is most faceted.
    const mf::Vector3 directions[] = {
        mf::Vector3(1.0, 0.0, 0.0), mf::Vector3(0.0, 1.0, 0.0), mf::Vector3(0.0, 0.0, 1.0),
        mf::Vector3(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
        mf::Vector3(0.7071067811865476, 0.7071067811865476, 0.0),
    };
    for (const mf::Vector3& direction : directions) {
        for (double offset : {-0.10, -0.05, 0.05, 0.10}) {
            const mf::Point3 x = (baked.radius + offset) * direction.normalized();
            const double actual = mean->evaluate(x).value;
            worst = std::max(worst, std::abs(actual - offset));
            ++sampled;
        }
    }
    context.require(sampled > 0, "sampled the band along several directions");
    context.require(worst <= tolerance,
                    "baked sdf tracks the analytic sphere (worst error " +
                        std::to_string(worst) + ", tolerance " + std::to_string(tolerance) + ")");

    // Sign, which is the part a magnitude-only check would miss.
    context.require(mean->evaluate(mf::Point3(1.10, 0.0, 0.0)).value > 0.0,
                    "a point outside the sphere has a positive sdf");
    context.require(mean->evaluate(mf::Point3(0.90, 0.0, 0.0)).value < 0.0,
                    "a point inside the sphere has a negative sdf");
    // Monotone across the surface, which catches an axis swap or a flipped map.
    double previous = -1e30;
    bool monotone = true;
    for (double radius = 0.90; radius <= 1.1001; radius += 0.01) {
        const double value = mean->evaluate(mf::Point3(radius, 0.0, 0.0)).value;
        if (value < previous) { monotone = false; break; }
        previous = value;
    }
    context.require(monotone, "the sdf increases monotonically through the surface");
}

// The gradient the renderer consumes must be the gradient of the function the
// renderer evaluates -- the trilinear interpolant, not an analytic ideal.
void testGradientMatchesCentralDifference(TestContext& context) {
    const BakedSphere& baked = sphereField();
    auto mean = mf::NanoVdbMean::open(baked.path);
    const double h = baked.report.dx * 0.25;

    double worst = 0.0;
    const mf::Point3 points[] = {mf::Point3(1.05, 0.02, 0.03), mf::Point3(0.96, -0.04, 0.02),
                                 mf::Point3(0.0, 1.07, 0.01), mf::Point3(0.6, 0.6, 0.6)};
    for (const mf::Point3& x : points) {
        const mf::MeanJet jet = mean->evaluate(x);
        for (int axis = 0; axis < 3; ++axis) {
            mf::Point3 plus = x;
            mf::Point3 minus = x;
            plus[axis] += h;
            minus[axis] -= h;
            const double finite =
                (mean->evaluate(plus).value - mean->evaluate(minus).value) / (2.0 * h);
            worst = std::max(worst, std::abs(finite - jet.gradient[axis]));
        }
    }
    // The interpolant is piecewise trilinear, so a central difference straddling
    // a cell boundary differs from the analytic derivative at the sample point.
    context.require(worst < 0.5,
                    "analytic gradient agrees with central differences (worst " +
                        std::to_string(worst) + ")");
}

// classicMajorant and the optical-depth sampler both call bounds() per flight,
// so it must be cheap and must never under-estimate.
void testBoundsAreConservative(TestContext& context) {
    const BakedSphere& baked = sphereField();
    auto mean = mf::NanoVdbMean::open(baked.path);
    auto alphaField = mf::NanoVdbSampledField::open(baked.path, "alpha");

    const mf::Bounds3 whole = mean->gridBounds();
    const mf::BoundsSummary summary = mean->bounds(whole);
    context.require(summary.certified, "the baked mean reports certified bounds");
    // The band is written for |sdf| strictly inside 3 sigma, so the innermost
    // stored value approaches -3 sigma from above and never crosses it, while a
    // band that stopped early would leave a visibly shallower minimum.
    const double inner = -baked.settings.bandSigmas * baked.settings.sigma;
    context.require(summary.minimumValue >= inner - 1e-9,
                    "the sdf never dips below the inner edge of the band");
    context.require(summary.minimumValue <= inner + baked.report.dx,
                    "the sdf reaches the inner edge of the band (min " +
                        std::to_string(summary.minimumValue) + " vs " + std::to_string(inner) + ")");
    context.require(summary.maximumGradientNorm > 0.0,
                    "the gradient bound is a positive number");

    // Measured maximum |grad| inside the band must sit under the bound.
    double measured = 0.0;
    for (double radius = 0.88; radius <= 1.12; radius += 0.02) {
        const mf::MeanJet jet = mean->evaluate(mf::Point3(radius, 0.0, 0.0));
        measured = std::max(measured, jet.gradient.norm());
    }
    context.require(measured <= summary.maximumGradientNorm,
                    "the gradient bound dominates the measured maximum (" +
                        std::to_string(measured) + " <= " +
                        std::to_string(summary.maximumGradientNorm) + ")");

    // A domain strictly inside the grid must not report a narrower range than
    // the samples inside it -- bounds() is allowed to be loose, never tight.
    mf::Bounds3 subdomain;
    subdomain.minimum = mf::Point3(-0.5, -0.5, -0.5);
    subdomain.maximum = mf::Point3(0.5, 0.5, 0.5);
    const mf::BoundsSummary sub = mean->bounds(subdomain);
    context.require(sub.minimumValue <= summary.minimumValue + 1e-12,
                    "a sub-domain reports at least the global minimum");
    context.require(sub.maximumGradientNorm >= measured,
                    "a sub-domain's gradient bound still dominates the samples in it");

    const mf::ScalarBounds alphaBounds = alphaField->bounds(whole);
    context.require(alphaBounds.certified, "the alpha field reports certified bounds");
    context.near(alphaBounds.maximumValue, baked.settings.alpha, 1e-6,
                 "alpha's upper bound is the configured alpha");
    context.require(alphaBounds.minimumValue > 0.0,
                    "alpha is strictly positive, as materialNdf requires");
}

void testTrilinearGradientNormBound(TestContext& context) {
    // In the cell [0,1]^3, the corner at the origin is zero and every other
    // corner is one. Near the origin all three partial derivatives approach
    // 1/dx, so the gradient norm exceeds the old range/dx bound.
    class CornerRampMean final : public mf::MeanField {
    public:
        const char* typeName() const override { return "test_corner_ramp"; }
        mf::MeanJet evaluate(const mf::Point3& x) const override {
            const mf::Point3 c = x.cwiseMax(0.0).cwiseMin(1.0);
            return {1.0 - (1.0 - c.x()) * (1.0 - c.y()) * (1.0 - c.z()),
                    mf::Vector3::Zero()};
        }
        mf::BoundsSummary bounds(const mf::Bounds3&) const override {
            return {0.0, std::sqrt(3.0), true};
        }
    } mean;

    mf::BakeSettings settings;
    settings.sigma = 1.0 / 6.0;  // SDF background is exactly the range maximum, 1.
    settings.fullDomain = true;
    const mf::Bounds3 domain{mf::Point3::Zero(), mf::Point3::Ones()};
    const mf::BakedGrids baked = mf::bakeAnalyticMean(
        mean, domain, 1.0, settings.sigma, settings.alpha);
    const auto path = tempFieldPath("trilinear_corner_gradient.nvdb");
    mf::writeFieldFile(path, baked, settings, "<corner ramp>");
    const auto sampled = mf::NanoVdbMean::open(path);
    const double measured = sampled->evaluate(mf::Point3::Constant(0.01)).gradient.norm();
    const double bound = sampled->bounds(sampled->gridBounds()).maximumGradientNorm;
    context.require(measured > 1.0 && measured <= bound,
                    "three simultaneous trilinear derivatives stay below the norm bound");
    context.near(bound, std::sqrt(3.0), 1e-6,
                 "the grid range bounds all three gradient components");
    context.near(baked.report.maximumGradientNorm, bound, 1e-6,
                 "analytic bake report agrees with the tracing bound");
    const auto sidecar = mf::readSidecar(path);
    context.require(sidecar.has_value(), "gradient test bake writes a sidecar");
    if (sidecar) context.near(sidecar->maximumGradientNorm, bound, 1e-6,
                              "sidecar records the tracing bound");
}

// The whole point of the factory: a config naming mean_type "nanovdb" builds the
// baked field, and the alpha grid travels with it.
void testFactoryRegistration(TestContext& context) {
    mf::registerNanoVdbFieldTypes();
    const BakedSphere& baked = sphereField();

    nlohmann::json fieldJson;
    fieldJson["mean_type"] = "nanovdb";
    fieldJson["grid_file"] = baked.path.string();

    const mf::MeanBuildResult built = mf::buildMeanFromJson(fieldJson);
    context.require(built.mean != nullptr, "the nanovdb mean type builds a field");
    context.require(built.alphaField != nullptr, "the nanovdb mean type supplies the alpha grid");

    auto typed = std::dynamic_pointer_cast<const mf::NanoVdbMean>(built.mean);
    context.require(typed != nullptr, "the built mean is a NanoVdbMean");
    if (typed) {
        context.near(typed->sigma(), baked.settings.sigma, 1e-6 * baked.settings.sigma,
                     "the built mean carries the baked sigma");
    }

    // An unknown type must be named in the error, since that is what a user sees
    // when a config has a typo.
    nlohmann::json bad;
    bad["mean_type"] = "no_such_field";
    bool threw = false;
    try {
        mf::buildMeanFromJson(bad);
    } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).find("no_such_field") != std::string::npos;
    }
    context.require(threw, "an unknown mean type throws and names the type");

    // The procedural types must still build: the registry replaced an if/else
    // chain, and a regression there would break every existing config.
    nlohmann::json sphere;
    sphere["mean_type"] = "sphere";
    sphere["sphere_center"] = {0.0, 0.0, 0.0};
    sphere["sphere_radius"] = 1.0;
    const mf::MeanBuildResult procedural = mf::buildMeanFromJson(sphere);
    context.require(procedural.mean != nullptr, "the procedural sphere type still builds");
    context.require(procedural.alphaField == nullptr,
                    "a procedural mean supplies no alpha field");
}

// A sigma that disagrees with the file is a silent rendering bug otherwise, so
// both explicit override and sidecar mismatch must fail loudly.
void testSigmaDisagreementThrows(TestContext& context) {
    const BakedSphere& baked = sphereField();

    bool threw = false;
    try {
        mf::NanoVdbMean::open(baked.path, baked.settings.sigma * 2.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    context.require(threw, "a sigma override that disagrees with the file is rejected");

    // A matching override is accepted.
    bool accepted = true;
    try {
        mf::NanoVdbMean::open(baked.path, baked.settings.sigma);
    } catch (const std::exception&) {
        accepted = false;
    }
    context.require(accepted, "a sigma override that matches the file is accepted");

    // Truncating the sidecar must not silently change sigma.
    const std::filesystem::path corrupt = tempFieldPath("corrupt_sidecar.nvdb");
    std::filesystem::copy_file(baked.path, corrupt,
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ifstream existing(baked.path.string() + ".json");
        nlohmann::json document;
        existing >> document;
        document["sigma"] = baked.settings.sigma * 3.0;
        std::ofstream out(corrupt.string() + ".json");
        out << document;
    }
    bool mismatchThrew = false;
    try {
        mf::resolveSigma(corrupt);
    } catch (const std::runtime_error&) {
        mismatchThrew = true;
    }
    context.require(mismatchThrew, "a sidecar that disagrees with the sigma grid is rejected");
}

// loadExperimentConfig installs the config's numeric policy globally; the
// throwaway configs below carry test-sized budgets, so the suite's policy has to
// come back.
struct RestoreNumericPolicy {
    mf::NumericPolicy saved = mf::defaultNumericPolicy();
    ~RestoreNumericPolicy() { mf::defaultNumericPolicy() = saved; }
};

// A complete config for the baked sphere with "sigma" deliberately absent --
// which is the point: a baked field's schema has no sigma key, since the file
// carries it. Only the keys loadExperimentConfig actually reads are here,
// because one it ignores would imply a contract it does not have.
nlohmann::json bakedConfigJson() {
    return nlohmann::json::parse(R"json({
        "field": {
            "mean_type": "nanovdb",
            "use_alpha_grid": true,
            "correlation_lengths": [0.1, 0.1, 0.1],
            "kernel_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ndf_family": "generalized_gaussian"
        },
        "material": {"eta_rgb": [0.2, 0.9, 1.1], "k_rgb": [3.9, 2.5, 2.2]},
        "transport": {},
        "fixed_flight": {
            "birth_position": [0, 0, 1], "direction": [0, 0, 1],
            "requested_maximum_age": 0.3, "curve_sample_count": 3, "flight_sample_count": 4
        },
        "numeric": {
            "relative_tolerance": 0.0001, "absolute_tolerance": 0.0000001,
            "max_quadrature_subdivisions": 128, "max_root_iterations": 64
        },
        "render": {
            "width": 2, "height": 2, "samples_per_pixel": 1,
            "camera_position": [0, -2, 1], "camera_target": [0, 0, 0],
            "vertical_fov_degrees": 45, "environment": "directional_gradient"
        },
        "output_directory": "unused_sigma_schema_test_output"
    })json");
}

// The two field schemas differ on exactly one key, and the difference is that a
// baked field has no free scale: sigma is part of its data, so the config may
// omit it and the file supplies it. A procedural mean is scale-free, so its
// config must state one -- nothing else could. Both directions are checked here,
// plus the two ways a contradicting sigma can still arrive (a config that states
// one, and --sigma), because either silently reinterprets the band and the
// background at the wrong scale and renders a plausible but wrong surface.
void testSigmaSchema(TestContext& context) {
    RestoreNumericPolicy restorePolicy;
    mf::registerNanoVdbFieldTypes();
    const BakedSphere& baked = sphereField();
    const double sigma = baked.settings.sigma;

    // The capability the loader consults, on the field itself.
    const auto mean = mf::NanoVdbMean::open(baked.path);
    context.require(mean->intrinsicSigma().has_value(),
                    "a baked mean declares that it fixes its own sigma");
    context.near(*mean->intrinsicSigma(), sigma, 1e-6 * sigma,
                 "the declared sigma is the one baked into the file");
    context.require(!mf::buildDefaultField().mean->intrinsicSigma().has_value(),
                    "a procedural mean declares no sigma of its own");

    const std::filesystem::path configPath = tempFieldPath("sigma_schema_config.json");
    auto load = [&](const nlohmann::json& value, std::string& error)
            -> std::optional<mf::ExperimentConfig> {
        {
            std::ofstream stream(configPath);
            stream << value.dump();
        }
        error.clear();
        try {
            return mf::loadExperimentConfig(configPath);
        } catch (const std::exception& thrown) {
            error = thrown.what();
            return std::nullopt;
        }
    };

    nlohmann::json config = bakedConfigJson();
    // Assigned rather than substituted into the JSON text: nlohmann escapes the
    // Windows separators, and a raw path would not survive parsing.
    config["field"]["grid_file"] = baked.path.string();

    // The requested behaviour: a baked config does not have to state a sigma.
    std::string error;
    const auto omitted = load(config, error);
    context.require(omitted.has_value(), "a baked config may omit field.sigma: " + error);
    if (omitted) {
        context.near(omitted->field.kernel.sigma(), sigma, 1e-7 * sigma,
                     "an omitted sigma resolves to the file's");
        const auto bakedMean = std::dynamic_pointer_cast<const mf::NanoVdbMean>(
            omitted->field.mean);
        context.require(bakedMean &&
                        omitted->field.activeDomain.contains(bakedMean->gridBounds().minimum) &&
                        omitted->field.activeDomain.contains(bakedMean->gridBounds().maximum),
                        "a baked field derives its tracing domain from its grid");
    }

    nlohmann::json staleDomain = config;
    staleDomain["field"]["domain_min"] = {-0.1, -0.1, -0.1};
    staleDomain["field"]["domain_max"] = {0.1, 0.1, 0.1};
    const auto ignoredDomain = load(staleDomain, error);
    context.require(ignoredDomain.has_value(), "legacy NVDB domain keys do not block loading: " + error);
    if (omitted && ignoredDomain) {
        context.require((omitted->field.activeDomain.minimum -
                         ignoredDomain->field.activeDomain.minimum).norm() == 0.0 &&
                        (omitted->field.activeDomain.maximum -
                         ignoredDomain->field.activeDomain.maximum).norm() == 0.0,
                        "stale NVDB domain keys cannot clip the imported grid");
    }

    // Restating it is still allowed and still means the same thing, which is what
    // keeps every config written before this schema split working. The agreement
    // is judged through a float32 grid, so it is not exact; the tolerance is what
    // separates that from a genuine re-bake.
    nlohmann::json restated = config;
    restated["field"]["sigma"] = sigma;
    const auto withSigma = load(restated, error);
    context.require(withSigma.has_value(),
                    "a baked config may restate the sigma it baked at: " + error);
    if (withSigma && omitted) {
        context.near(withSigma->field.kernel.sigma(), omitted->field.kernel.sigma(), 1e-9,
                     "restating the file's sigma resolves to the same kernel");
    }

    // A sigma that disagrees must not be quietly preferred over the file's.
    nlohmann::json wrong = config;
    wrong["field"]["sigma"] = sigma * 2.0;
    const auto mismatched = load(wrong, error);
    context.require(!mismatched.has_value(), "a baked config cannot state a different sigma");
    context.require(error.find("sigma") != std::string::npos,
                    "the rejection names sigma: " + error);

    // The procedural schema is unchanged: sigma stays required, and the error
    // says which type wanted one.
    nlohmann::json procedural = config;
    procedural["field"]["mean_type"] = "sphere";
    procedural["field"]["sphere_center"] = {0.0, 0.0, 0.0};
    procedural["field"]["sphere_radius"] = 1.0;
    procedural["field"].erase("grid_file");
    procedural["field"].erase("use_alpha_grid");
    const auto missing = load(procedural, error);
    context.require(!missing.has_value(), "a procedural config still requires field.sigma");
    context.require(error.find("sphere") != std::string::npos,
                    "the rejection names the mean type that needs it: " + error);
    procedural["field"]["sigma"] = sigma;
    const auto noDomain = load(procedural, error);
    context.require(!noDomain.has_value() && error.find("domain_min") != std::string::npos,
                    "procedural fields still require an explicit tracing domain");
    procedural["field"]["domain_min"] = {-1.6, -1.6, -1.6};
    procedural["field"]["domain_max"] = {1.6, 1.6, 1.6};
    context.require(load(procedural, error).has_value(),
                    "a procedural config that states a sigma still loads: " + error);

    // --sigma is the other way a contradicting value can reach a baked field.
    if (omitted) {
        mf::ExperimentConfig overridden = *omitted;
        bool rejected = false;
        try {
            mf::applyFieldOverrides(overridden, sigma * 2.0, std::nullopt, false);
        } catch (const std::invalid_argument& thrown) {
            rejected = std::string(thrown.what()).find("re-bake") != std::string::npos;
        }
        context.require(rejected, "--sigma cannot override a baked field's sigma");

        // An agreeing value is a no-op rather than an error, so a sweep script
        // that passes one sigma to every config keeps working.
        mf::ExperimentConfig agreeing = *omitted;
        bool threw = false;
        try {
            mf::applyFieldOverrides(agreeing, sigma, std::nullopt, false);
        } catch (const std::exception&) {
            threw = true;
        }
        context.require(!threw, "--sigma may restate a baked field's sigma");
        // Exactly, not approximately: the point of the rule is that a baked field
        // has one sigma, the file's, so an agreeing override must not swap in the
        // config's spelling of it one float32 ulp away.
        context.require(agreeing.field.kernel.sigma() == omitted->field.kernel.sigma(),
                        "restating via --sigma leaves the kernel bit-identical");
    }

    // The resolved record states the sigma the run actually used, even though the
    // input omitted it: that is what makes the output reproducible.
    if (omitted) {
        mf::writeResolvedConfig(*omitted, configPath);
        std::ifstream stream(configPath);
        nlohmann::json resolved;
        stream >> resolved;
        context.near(resolved.at("field").at("sigma").get<double>(), sigma, 1e-6 * sigma,
                     "the resolved config records the sigma taken from the file");
        context.near(resolved.at("derived").at("sigma").get<double>(), sigma, 1e-7 * sigma,
                     "the resolved config's derived sigma is the same one");
    }

    // The tolerance the two agreement checks above ride on. A sigma written by the
    // generator and read back through the float32 sigma grid is the same sigma at
    // every value a user might pick -- an absolute 1e-9 window called most of them
    // different (0.06, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3), which made the
    // generator's own output unreadable at those sigmas.
    for (double candidate : {0.01, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0}) {
        const double stored = static_cast<double>(static_cast<float>(candidate));
        context.require(mf::sameSigma(candidate, stored),
                        "a sigma survives the float32 round trip at " + std::to_string(candidate));
        context.require(!mf::sameSigma(candidate, candidate * 1.01),
                        "a one-percent difference is not the same sigma at " +
                            std::to_string(candidate));
        context.require(!mf::sameSigma(candidate, candidate * 2.0),
                        "a doubled sigma is not the same sigma at " + std::to_string(candidate));
    }

    // The wiring behind that tolerance, at the value that exposes it: a bake at
    // sigma = 0.1 is the generator's own output, and the loader that reads it is
    // the one this schema hands the field to.
    mf::BakeSettings settings;
    settings.sigma = 0.1;
    settings.resolution = {24, 24, 24};
    const std::filesystem::path roundTrip = tempFieldPath("sigma_round_trip.nvdb");
    mf::writeFieldFile(roundTrip, mf::bakeMacrofacetField(mf::makeSphereMesh(1.0, 48), settings),
                       settings, "<sigma round trip>");
    context.near(mf::resolveSigma(roundTrip), 0.1, 1e-6,
                 "a sigma = 0.1 bake resolves its own sigma");
    bool opened = true;
    try {
        mf::NanoVdbMean::open(roundTrip);
    } catch (const std::exception&) {
        opened = false;
    }
    context.require(opened, "a sigma = 0.1 field opens without anything restating its sigma");
}

// A mesh far from the origin exercises the map: a wrong origin or a half-voxel
// offset shows up as a constant error everywhere, which the centred sphere above
// cannot distinguish from a wrong radius.
void testOffCentreMesh(TestContext& context) {
    mf::BakeSettings settings;
    settings.sigma = 0.05;
    settings.resolution = {32, 32, 32};
    const double radius = 1.0;
    const mf::Vector3 centre(5.0, -3.0, 2.0);

    mf::TriangleMesh mesh = mf::makeSphereMesh(radius, 128);
    mesh.vertices.rowwise() += centre.transpose();

    const std::filesystem::path path = tempFieldPath("sphere_offset.nvdb");
    mf::BakedGrids grids = mf::bakeMacrofacetField(mesh, settings);
    // The frame must cover the object, not the origin.
    context.near(grids.report.origin.x(), centre.x() - radius - 3.0 * settings.sigma, 1e-9,
                 "the grid origin tracks the mesh, not the world origin");
    mf::writeFieldFile(path, grids, settings, "<offset sphere>");

    auto mean = mf::NanoVdbMean::open(path);
    const mf::Vector3 axis = mf::Vector3(1.0, 0.0, 0.0);
    const double outside = mean->evaluate(centre + (radius + 0.08) * axis).value;
    const double inside = mean->evaluate(centre + (radius - 0.08) * axis).value;
    context.near(outside, 0.08, facetedSphereSagitta(radius, 128) + 1e-2,
                 "an off-centre mesh still has the right sdf outside");
    context.near(inside, -0.08, facetedSphereSagitta(radius, 128) + 1e-2,
                 "an off-centre mesh still has the right sdf inside");
}

// Builds the render-side field from the baked file: the sdf grid as the mean, the
// alpha grid as the material. This is exactly what a config with mean_type
// "nanovdb" produces, minus the JSON.
mf::GPSSField bakedRenderField(const BakedSphere& baked) {
    mf::GPSSField field;
    field.mean = mf::NanoVdbMean::open(baked.path);
    // The same precision the procedural configs use: P = (roughness/sigma)^2 with
    // roughness = sigma, i.e. the identity.
    field.kernel = mf::SquaredExponentialKernel(baked.settings.sigma, mf::Matrix3::Identity());
    field.activeDomain.minimum = mf::Point3::Constant(-1.6);
    field.activeDomain.maximum = mf::Point3::Constant(1.6);
    return field;
}

// materialNdf() is the split between the two roles covarianceG carries. With an
// alpha grid the material covariance is alpha^2 I and the transport covariance
// must be untouched; with no alpha grid the two must be the same object's worth
// of numbers, which is what keeps every procedural config bit-identical.
void testMaterialNdfRoleSeparation(TestContext& context) {
    const BakedSphere& baked = sphereField();
    const double sigma = baked.settings.sigma;
    const double alpha = baked.settings.alpha;

    const mf::GPSSField field = bakedRenderField(baked);
    mf::MaterialConfig materialWithAlpha;
    materialWithAlpha.alphaField = mf::NanoVdbSampledField::open(baked.path, "alpha");
    mf::MaterialConfig materialWithout;

    // Sample where the bake is meaningful: on the surface.
    const mf::Point3 x(0.0, 0.0, baked.radius);
    const mf::PointPrior transport = field.pointPrior(x);
    const mf::PointPrior material = materialWithAlpha.materialNdf(field, x);

    const mf::Matrix3 expected = (alpha * alpha) * mf::Matrix3::Identity();
    context.require((material.covarianceG - expected).norm() <= 1e-9,
                    "materialNdf carries alpha^2 I as the material covariance");

    // The transport covariance must NOT have been replaced. Without this the
    // alpha grid would silently rewrite the GP statistics.
    const mf::Matrix3 transportExpected = sigma * sigma * field.kernel.precision();
    context.require((transport.covarianceG - transportExpected).norm() <= 1e-9,
                    "pointPrior still carries sigma^2 P under an alpha grid");

    // The observation components pass through untouched, because
    // evaluateClassic reads meanF for the density term from the same prior.
    context.near(material.meanF, transport.meanF, 1e-12,
                 "materialNdf leaves the mean value alone");
    context.require((material.meanG - transport.meanG).norm() <= 1e-12,
                    "materialNdf leaves the mean gradient alone");

    // ggxAlphaAt is the GGX family's route to the same grid: scalar lifted to
    // isotropic. Without a grid it must hand back the config's anisotropy
    // untouched, or every existing GGX config silently becomes isotropic.
    context.require((materialWithAlpha.ggxAlphaAt(x) - mf::Vector2::Constant(alpha)).norm() <= 1e-9,
                    "ggxAlphaAt reads the alpha grid as an isotropic alpha");
    const mf::Vector2 anisotropic(0.2, 0.7);
    mf::MaterialConfig ggxOnly = materialWithout;
    ggxOnly.ggxAlpha = anisotropic;
    context.require((ggxOnly.ggxAlphaAt(x) - anisotropic).norm() == 0.0,
                    "ggxAlphaAt returns the config's anisotropic alpha when there is no grid");

    // No alpha grid: pointPrior verbatim, which is the no-regression guarantee.
    const mf::PointPrior plainTransport = field.pointPrior(x);
    const mf::PointPrior plainMaterial = materialWithout.materialNdf(field, x);
    context.near(plainMaterial.meanF, plainTransport.meanF, 1e-15,
                 "without an alpha grid materialNdf equals pointPrior (meanF)");
    context.require((plainMaterial.covarianceG - plainTransport.covarianceG).norm() == 0.0,
                    "without an alpha grid materialNdf equals pointPrior (covarianceG)");
    context.require((plainMaterial.meanG - plainTransport.meanG).norm() == 0.0,
                    "without an alpha grid materialNdf equals pointPrior (meanG)");
}

// classicMajorant's contract: for every point in the domain, evaluateClassic's
// extinction must not exceed it. This is not a conservativeness nicety -- the
// tracker bounds the hazard by the majorant, so one that sits below the true
// value drops real collisions and renders as missing geometry.
//
// Alpha reaches the extinction only through the GGX family, and there the
// projected-area formula grows with alpha. A majorant built from the config's
// ggxAlpha therefore goes below the true value wherever alpha(x) exceeds it.
// This is the one assertion in the suite that fails if the majorant reads
// ggxAlpha instead of the grid's domain maximum.
void testMajorantCoversAlphaField(TestContext& context) {
    const BakedSphere& baked = sphereField();
    const double alpha = baked.settings.alpha;

    mf::GPSSField field = bakedRenderField(baked);
    mf::MaterialConfig material;
    material.alphaField = mf::NanoVdbSampledField::open(baked.path, "alpha");
    material.ndfFamily = mf::NdfFamily::GGXBaseline;
    // Deliberately below the grid's alpha, so the config value alone is not a
    // bound. A config would hit this by leaving ggx_alpha at its default.
    material.ggxAlpha = mf::Vector2(0.05, 0.05);
    // No gradient, so the majorant's gradient term contributes nothing and the
    // alpha term is the whole story. (On the real baked sdf the (max-min)/dx
    // gradient bound is ~1/dx and would swamp it, making any bound pass.)
    field.mean = std::make_shared<mf::ConstantMean>(0.0);

    const mf::Vector3 w = mf::Vector3(0.3, -0.5, 0.8).normalized();
    const double majorant = mf::classicMajorant(field, material, field.activeDomain, w);
    context.require(majorant > 0.0 && std::isfinite(majorant), "the majorant is positive");

    double worstRatio = 0.0;
    int checked = 0;
    for (int i = 0; i <= 16; ++i) {
        for (int j = 0; j <= 16; ++j) {
            const mf::Point3 x(-1.0 + 2.0 * i / 16.0, -1.0 + 2.0 * j / 16.0, 0.5);
            const mf::ClassicEvaluation value = mf::evaluateClassic(field, material, x, w);
            if (value.extinction.status != mf::NumericStatus::Ok) continue;
            ++checked;
            worstRatio = std::max(worstRatio, value.extinction.value / majorant);
        }
    }
    context.require(checked > 0, "the majorant sweep actually evaluated points");
    // A few ulps of slack: evaluateClassic multiplies in log space while the
    // majorant does not, so the two products can round apart by an ulp or two.
    context.require(worstRatio <= 1.0 + 1e-12,
                    "the GGX majorant bounds the extinction under an alpha grid (worst ratio " +
                        std::to_string(worstRatio) + ")");

    // And the config's own ggxAlpha would NOT have covered the same sweep --
    // that is exactly the bug the domain maximum exists to prevent. Rebuild the
    // bound from the config alpha, changing nothing else.
    const mf::Vector3 wn = w.normalized();
    const auto ggxArea = [&wn](double a) {
        return 0.5 * (std::sqrt(wn.z() * wn.z() + a * a * wn.x() * wn.x() +
                                a * a * wn.y() * wn.y()) -
                      wn.z());
    };
    const double gridArea = ggxArea(alpha);
    const double configArea = ggxArea(material.ggxAlpha.x());
    context.require(configArea > 0.0 && configArea < gridArea,
                    "the config alpha is below the grid alpha, so this sweep is a real test");

    const double gridMajorant = majorant;
    const double configMajorant = gridMajorant * (configArea / gridArea);
    const double worstValue = worstRatio * gridMajorant;
    context.require(configMajorant < worstValue,
                    "the config-alpha bound is exceeded (" + std::to_string(configMajorant) +
                        " < " + std::to_string(worstValue) +
                        "), so the domain maximum is what makes the GGX majorant valid");
}

// The same contract on the production path: the baked field the renderer actually
// loads, swept along two shells around the surface.
void testMajorantCoversBakedField(TestContext& context) {
    const BakedSphere& baked = sphereField();
    const double sigma = baked.settings.sigma;

    const mf::GPSSField field = bakedRenderField(baked);
    mf::MaterialConfig material;
    material.alphaField = mf::NanoVdbSampledField::open(baked.path, "alpha");
    const mf::Vector3 w = mf::Vector3(0.3, -0.5, 0.8).normalized();
    const double majorant = mf::classicMajorant(field, material, field.activeDomain, w);
    context.require(majorant > 0.0 && std::isfinite(majorant), "the baked majorant is positive");

    double worstRatio = 0.0;
    int checked = 0;
    const int steps = 16;
    for (int i = 0; i <= steps; ++i) {
        for (int j = 0; j <= steps; ++j) {
            mf::Point3 x(-1.0 + 2.0 * i / steps, -1.0 + 2.0 * j / steps, 0.0);
            const double rho = x.norm();
            if (!(rho > 0.0)) continue;
            for (double radius : {baked.radius, baked.radius - 2.0 * sigma}) {
                const mf::ClassicEvaluation value =
                    mf::evaluateClassic(field, material, mf::Point3(x * (radius / rho)), w);
                if (value.extinction.status != mf::NumericStatus::Ok) continue;
                ++checked;
                worstRatio = std::max(worstRatio, value.extinction.value / majorant);
            }
        }
    }
    context.require(checked > 0, "the baked majorant sweep actually evaluated points");
    context.require(worstRatio <= 1.0,
                    "the majorant bounds the extinction on the baked field (worst ratio " +
                        std::to_string(worstRatio) + ")");
}

// The transform is what makes an imported model usable: sigma is a world-space
// length, so a mesh in its author's units gets a +-3 sigma band that either
// vanishes or swallows the object. The arithmetic is three lines, but a wrong
// scale silently corrupts every field, so it is checked against a bake rather
// than only against itself.
void testMeshTransform(TestContext& context) {
    // The scale a downloaded model tends to arrive in: scenes/shader ball.ply is
    // 0.089 across, 22x smaller than the procedural ball it stands in for.
    const double authoredRadius = 0.0446;
    mf::TriangleMesh mesh = mf::makeSphereMesh(authoredRadius, 128);
    mesh.vertices.rowwise() += mf::Vector3(0.0, 0.0, 0.09).transpose();

    // An unrequested transform must be the identity bit for bit: every existing
    // config bakes through this path, and none of them asked for a transform.
    const mf::MeshTransform none = mf::fitMeshTransform(mesh, 0.0, false);
    context.require(none.isIdentity(), "no --fit/--center leaves the size and position alone");
    const Eigen::MatrixXd untouched = mesh.vertices;
    mf::applyMeshTransform(mesh, none);
    context.require((mesh.vertices - untouched).cwiseAbs().maxCoeff() == 0.0,
                    "the identity transform is bit-exact");

    // --fit 2.0 --center: a bounding box centred on the origin, 2.0 across.
    const mf::MeshTransform transform = mf::fitMeshTransform(mesh, 2.0, true);
    mf::applyMeshTransform(mesh, transform);
    const mf::Bounds3 bounds = mesh.bounds();
    context.near(0.5 * (bounds.minimum + bounds.maximum).norm(), 0.0, 1e-12,
                 "--center moves the bounding-box centre to the origin");
    context.near((bounds.maximum - bounds.minimum).maxCoeff(), 2.0, 1e-12,
                 "--fit makes the largest extent the requested one");

    // The decisive check: the fitted mesh bakes the sdf of the unit sphere it has
    // become. Skipping the transform would leave the radius at 0.0446, and both
    // samples below would land deep inside the surface with the wrong sign.
    mf::BakeSettings settings;
    settings.sigma = 0.05;
    settings.resolution = {32, 32, 32};
    const std::filesystem::path path = tempFieldPath("sphere_fitted.nvdb");
    mf::BakedGrids grids = mf::bakeMacrofacetField(mesh, settings);
    mf::writeFieldFile(path, grids, settings, "<fitted sphere>", transform);

    auto mean = mf::NanoVdbMean::open(path);
    const mf::Vector3 axis = mf::Vector3(1.0, 0.0, 0.0);
    const double outside = mean->evaluate(1.08 * axis).value;
    const double inside = mean->evaluate(0.92 * axis).value;
    context.near(outside, 0.08, facetedSphereSagitta(1.0, 128) + 1e-2,
                 "a fitted mesh bakes the sdf of the unit sphere it became");
    context.near(inside, -0.08, facetedSphereSagitta(1.0, 128) + 1e-2,
                 "a fitted mesh bakes the right sign inside");

    // Contract of the guard: a scale that would collapse or mirror the mesh, and
    // any non-finite component, are refused -- and refused without touching the
    // mesh. (A merely *small* scale is not refused: whether the band survives it
    // depends on sigma, which this function does not know; bandIsResolvable
    // answers that instead.)
    const Eigen::MatrixXd fitted = mesh.vertices;
    const mf::MeshTransform rejected[] = {
        mf::MeshTransform{0.0, mf::Vector3::Zero()},   // collapses to a point
        mf::MeshTransform{-2.0, mf::Vector3::Zero()},  // mirrors
        mf::MeshTransform{std::nan(""), mf::Vector3::Zero()},
        mf::MeshTransform{1.0, mf::Vector3::Constant(std::nan(""))},
    };
    for (const mf::MeshTransform& bad : rejected) {
        bool threw = false;
        try {
            mf::applyMeshTransform(mesh, bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        context.require(threw, "a collapsing or non-finite mesh transform is rejected");
    }
    context.require((mesh.vertices - fitted).cwiseAbs().maxCoeff() == 0.0,
                    "a rejected transform leaves the fitted mesh bit-exact");

    // The one case the positivity check cannot see: a scale small enough to
    // underflow every coordinate to zero. It has to be caught where the cause is
    // visible rather than later in makeFrame, and it must not leave the mesh
    // flattened on the way out.
    mf::TriangleMesh tiny = mesh;
    tiny.vertices *= 1e-300;
    const Eigen::MatrixXd tinyBefore = tiny.vertices;
    bool underflowed = false;
    try {
        mf::applyMeshTransform(tiny, mf::MeshTransform{1e-300, mf::Vector3::Zero()});
    } catch (const std::invalid_argument&) {
        underflowed = true;
    }
    context.require(underflowed, "a transform that underflows the mesh to a point is rejected");
    context.require((tiny.vertices - tinyBefore).cwiseAbs().maxCoeff() == 0.0,
                    "a rejected transform does not modify the mesh");
}

void testAnalyticFullDomainBake(TestContext& context) {
    using namespace mf;
    const SphereMean analytic(Point3::Zero(), 0.5);
    const Bounds3 domain{Point3::Constant(-1.0), Point3::Constant(1.0)};
    const std::filesystem::path path = tempFieldPath("analytic_full_domain.nvdb");
    const BakedGrids grids = bakeAnalyticMean(analytic, domain, 0.1, 0.1, 0.5);
    BakeSettings settings;
    settings.sigma = 0.1;
    settings.alpha = 0.5;
    settings.bandSigmas = 0.0;
    settings.fullDomain = true;
    writeFieldFile(path, grids, settings, "<analytic sphere>");
    nlohmann::json autoDomainConfig = bakedConfigJson();
    autoDomainConfig["field"]["grid_file"] = path.string();
    autoDomainConfig["fixed_flight"]["birth_position"] = {0.0, 0.0, 5.0};
    const std::filesystem::path configPath = tempFieldPath("auto_domain_config.json");
    {
        std::ofstream stream(configPath);
        stream << autoDomainConfig.dump();
    }
    ExperimentConfig prepared = loadExperimentConfig(configPath);
    context.require(!prepared.field.activeDomain.contains(prepared.fixedFlight.birthPosition),
                    "render config can load when its unused fixed-flight birth is outside the grid");
    prepareNanoVdbField(prepared);
    context.require(!prepared.mediumSurfaceBand,
                    "full-domain NVDB retains its coverage after automatic domain resolution");
    const auto density = NanoVdbSampledField::open(path, "density");
    const Vector3 densityMargin = Vector3::Constant(density->voxelSize());
    context.require(prepared.field.activeDomain.contains(
                        density->gridBounds().minimum - densityMargin) &&
                    prepared.field.activeDomain.contains(
                        density->gridBounds().maximum + densityMargin),
                    "automatic tracing domain contains the density interpolation support");
    writeResolvedConfig(prepared, configPath);
    {
        std::ifstream stream(configPath);
        nlohmann::json resolved;
        stream >> resolved;
        context.near(resolved["derived"]["domain_min"][0].get<double>(),
                     prepared.field.activeDomain.minimum.x(), 1e-12,
                     "resolved config records the automatic tracing domain");
    }
    auto sampled = NanoVdbMean::open(path);
    context.near(sampled->evaluate(Point3::Zero()).value, -0.5, 1e-5,
                 "full-domain bake preserves the sphere's negative core");
    context.near(sampled->evaluate(Point3(0.8, 0.0, 0.0)).value, 0.3, 0.01,
                 "full-domain bake preserves exterior distance");
    context.near(sampled->evaluate(Point3(1.0, 0.0, 0.0)).value, 0.5, 0.01,
                 "full-domain bake covers the active-domain boundary");

    GPSSField field;
    field.mean = sampled;
    field.kernel = SquaredExponentialKernel::fromCorrelationLengths(
        0.1, Vector3::Constant(0.25));
    field.activeDomain = domain;
    field.validate();
    ExperimentConfig experiment;
    experiment.field = field;
    experiment.mediumDensity = NanoVdbSampledField::open(path, "density");
    requireNanoVdbField(experiment);
    experiment.field.mean = std::make_shared<SphereMean>(Point3::Zero(), 0.5);
    bool rejectedProceduralTrace = false;
    try { requireNanoVdbField(experiment); }
    catch (const std::invalid_argument&) { rejectedProceduralTrace = true; }
    context.require(rejectedProceduralTrace,
                    "experiment tracing rejects an unbaked procedural mean");
    const FlightState state = startExternalFlight(Point3(0.5, 0.0, 0.0),
                                                   Vector3::UnitX());
    NarrowBandMedium medium(field, experiment.material);
    const auto flight = medium.beginFlight(state);
    const auto hazard = flight->evaluate(0.1).hazard.value;
    context.require(std::isfinite(hazard) && hazard >= 0.0,
                    "Classic evaluates the baked field");
}

void testMeshFullDomainBake(TestContext& context) {
    using namespace mf;
    const TriangleMesh mesh = makeSphereMesh(0.5, 32);
    BakeSettings settings;
    settings.sigma = 0.1;
    settings.resolution = {16, 16, 16};
    settings.fullDomain = true;
    const std::filesystem::path path = tempFieldPath("mesh_full_domain.nvdb");
    const BakedGrids grids = bakeMacrofacetField(mesh, settings);
    writeFieldFile(path, grids, settings, "<full-domain mesh sphere>");
    const auto sampled = NanoVdbMean::open(path);
    context.require(sampled->evaluate(Point3::Zero()).value < -0.3,
                    "full-domain mesh bake preserves the negative core");
    const auto sidecar = readSidecar(path);
    context.require(sidecar && sidecar->fullDomain,
                    "mesh field records its full-domain coverage");
}

void testNarrowBandTransport(TestContext& context) {
    using namespace mf;
    const BakedSphere& baked = sphereField();
    nlohmann::json importConfig = bakedConfigJson();
    importConfig["field"]["grid_file"] = baked.path.string();
    importConfig["fixed_flight"]["birth_position"] = {0.0, 0.0, 5.0};
    const std::filesystem::path configPath = tempFieldPath("auto_narrow_domain.json");
    {
        std::ofstream stream(configPath);
        stream << importConfig.dump();
    }
    ExperimentConfig prepared = loadExperimentConfig(configPath);
    prepareNanoVdbField(prepared);
    context.require(prepared.mediumSurfaceBand && prepared.densityMajorantGrid != nullptr,
                    "automatic narrow-band domain builds its DDA majorant grid");
    const auto importedDensity = NanoVdbSampledField::open(baked.path, "density");
    const Vector3 margin = Vector3::Constant(importedDensity->voxelSize());
    context.require(prepared.field.activeDomain.contains(
                        importedDensity->gridBounds().minimum - margin) &&
                    prepared.field.activeDomain.contains(
                        importedDensity->gridBounds().maximum + margin),
                    "automatic narrow-band domain contains density interpolation support");
    const GPSSField field = bakedRenderField(baked);
    MaterialConfig material;
    material.alphaField = NanoVdbSampledField::open(baked.path, "alpha");
    const ScalarFieldPtr density = NanoVdbSampledField::open(baked.path, "density");
    const auto majorants = std::make_shared<DensityMajorantGrid>(
        *density, field.activeDomain, 16);
    NarrowBandMedium medium(field, material, density, true, majorants);
    const Vector3 w = Vector3::UnitX();
    const FlightState external = startExternalFlight(Point3(-1.5, 0.0, 0.0), w);
    const auto segments = majorants->segments({external.birthPosition, w}, 0.0, 3.1);
    bool sawVacuum = false, sawBandAfterVacuum = false;
    for (const auto& segment : segments) {
        if (segment.densityMaximum == 0.0 && segment.begin > 0.7) sawVacuum = true;
        if (sawVacuum && segment.densityMaximum > 0.0) sawBandAfterVacuum = true;
    }
    context.require(sawVacuum && sawBandAfterVacuum,
                    "DDA skips vacuum macrocells and re-enters the far surface band");
    const auto classic = medium.beginFlight(external);
    context.require(classic->evaluate(0.5).hazard.value > 0.0 &&
                    classic->evaluate(1.5).hazard.value == 0.0 &&
                    classic->evaluate(2.5).hazard.value > 0.0,
                    "classic flight crosses a vacuum core and re-enters the surface band");
    Random rng(4821);
    for (int sample = 0; sample < 32; ++sample) {
        const FlightSample result = medium.sample(*classic, rng, defaultNumericPolicy(), nullptr, 16);
        context.require(result.age >= 0.0 && result.age <= classic->maximumAgeInDomain(),
                        "null tracking stays in the flight domain");
    }
    for (int sample = 0; sample < 16; ++sample) {
        const FlightSample result = medium.sample(*classic, rng, defaultNumericPolicy(),
                                                  nullptr, 16, 1.0);
        context.require(result.age <= 1.0 && (result.collided || result.age == 1.0),
                        "narrow-band medium respects a truncated flight interval");
    }
}

} // namespace

void testNanoVdbField(TestContext& context) {
    testVoxelContract(context);
    testBackgroundValues(context);
    testSignedDistanceAgreement(context);
    testGradientMatchesCentralDifference(context);
    testBoundsAreConservative(context);
    testTrilinearGradientNormBound(context);
    testFactoryRegistration(context);
    testSigmaDisagreementThrows(context);
    testSigmaSchema(context);
    testOffCentreMesh(context);
    testMaterialNdfRoleSeparation(context);
    testMajorantCoversAlphaField(context);
    testMajorantCoversBakedField(context);
    testMeshTransform(context);
    testAnalyticFullDomainBake(context);
    testMeshFullDomainBake(context);
    testNarrowBandTransport(context);
}

#include "macrofacet/fields/NanoVdbMean.h"

#include "fieldgen/NanoVdbIO.h"
#include "fields/NanoVdbGridSampler.h"
#include "fields/TrilinearBounds.h"
#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/fields/NanoVdbSampledField.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mf {

struct NanoVdbMean::Impl {
    detail::GridView view;
    std::filesystem::path path;
    double sigma = 0.0;

    // The global value range also bounds every interpolation cell. Apply the
    // three-dimensional trilinear gradient bound used by both bakers.
    double maximumGradientNorm() const {
        return detail::trilinearGradientNormBound(
            view.minimumValue(), view.maximumValue(), view.voxelSize());
    }
};

NanoVdbMean::NanoVdbMean() : impl_(std::make_unique<Impl>()) {}
NanoVdbMean::~NanoVdbMean() = default;

std::shared_ptr<const NanoVdbMean> NanoVdbMean::open(const std::filesystem::path& gridFile,
                                                     double sigmaOverride) {
    auto result = std::shared_ptr<NanoVdbMean>(new NanoVdbMean());
    result->impl_->path = gridFile;
    result->impl_->view = detail::GridView::open(gridFile, "sdf");

    const double fileSigma = resolveSigma(gridFile);
    if (sigmaOverride > 0.0) {
        // An explicit override still has to agree with the file: a mismatch
        // means the field was baked for a different statistical scale, which
        // renders as a plausible but wrong surface. Agreement is judged at the
        // precision the grid can store -- see sameSigma.
        if (!sameSigma(fileSigma, sigmaOverride)) {
            throw std::invalid_argument(
                "sigma override (" + std::to_string(sigmaOverride) +
                ") disagrees with the sigma stored in " + gridFile.string() + " (" +
                std::to_string(fileSigma) + "); re-bake or drop the override");
        }
    }
    result->impl_->sigma = fileSigma;

    const Bounds3& bounds = result->impl_->view.worldBounds();
    if (!bounds.valid()) {
        throw std::runtime_error("baked grid has a degenerate world bounding box: " +
                                 gridFile.string());
    }
    return result;
}

// The far-field value. At the background (+6 sigma) the hazard's density term
// is phi(6)/(sigma*Phi(6)) ~ 6.2e-9/sigma, so a ray travelling outside the grid
// effectively never collides -- which is also what makes scattering rays that
// miss the object terminate cleanly instead of picking up phantom hits.
MeanJet NanoVdbMean::evaluate(const Point3& x) const {
    MeanJet jet;
    impl_->view.sampleWithGradient(x, jet.value, jet.gradient);
    if (!std::isfinite(jet.value) || !jet.gradient.allFinite()) {
        // Only reachable through a corrupt grid; degrade to the vacuum value
        // rather than propagating NaNs into the transport algebra.
        jet.value = impl_->view.background();
        jet.gradient = Vector3::Zero();
    }
    return jet;
}

BoundsSummary NanoVdbMean::bounds(const Bounds3& domain) const {
    (void)domain;  // the grid's global extremes bound every sub-domain
    BoundsSummary summary;
    summary.minimumValue = impl_->view.minimumValue();
    summary.maximumGradientNorm = impl_->maximumGradientNorm();
    summary.certified = true;
    return summary;
}

double NanoVdbMean::voxelSizeHint() const { return impl_->view.voxelSize(); }

void NanoVdbMean::appendRayBreakpoints(const Point3& origin, const Vector3& direction,
                                       double begin, double end,
                                       std::vector<double>& knots) const {
    const double dx = impl_->view.voxelSize();
    const Point3& gridOrigin = impl_->view.origin();
    for (int axis = 0; axis < 3; ++axis) {
        const double speed = direction[axis];
        if (std::abs(speed) < 1e-14) continue;
        const double u0 = (origin[axis] + begin * speed - gridOrigin[axis]) / dx - 0.5;
        const double u1 = (origin[axis] + end * speed - gridOrigin[axis]) / dx - 0.5;
        const long long first = static_cast<long long>(std::floor(std::min(u0, u1))) + 1;
        const long long last = static_cast<long long>(std::ceil(std::max(u0, u1))) - 1;
        if (last - first > 100000) {
            throw std::invalid_argument("ray crosses too many NVDB interpolation cells");
        }
        for (long long index = first; index <= last; ++index) {
            const double coordinate = gridOrigin[axis] + (static_cast<double>(index) + 0.5) * dx;
            const double age = (coordinate - origin[axis]) / speed;
            if (age > begin && age < end) knots.push_back(age);
        }
    }
}

double NanoVdbMean::sigma() const { return impl_->sigma; }

// Exactly sigma(), offered through the interface a config can consult without
// knowing what a NanoVdbMean is.
std::optional<double> NanoVdbMean::intrinsicSigma() const { return sigma(); }

double NanoVdbMean::background() const { return impl_->view.background(); }
const std::filesystem::path& NanoVdbMean::gridFile() const { return impl_->path; }
const Bounds3& NanoVdbMean::gridBounds() const { return impl_->view.worldBounds(); }

namespace {

MeanBuildResult makeNanoVdbMean(const nlohmann::json& fieldJson) {
    const std::string file = fieldJson.at("grid_file").get<std::string>();
    const double sigmaOverride = fieldJson.value("sigma", 0.0);
    MeanBuildResult result;
    result.mean = NanoVdbMean::open(file, sigmaOverride);
    const auto& baked = static_cast<const NanoVdbMean&>(*result.mean);
    const Vector3 interpolationMargin = Vector3::Constant(baked.voxelSizeHint());
    result.activeDomain = {baked.gridBounds().minimum - interpolationMargin,
                           baked.gridBounds().maximum + interpolationMargin};
    // The alpha grid lives in the same file, so a config only ever names one
    // path. It is optional: the generator always writes it, but a file trimmed
    // down to the sdf grid stays usable (the material then falls back to the
    // config's roughness).
    if (fieldJson.value("use_alpha_grid", true) && nanovdb::io::hasGrid(file, "alpha")) {
        result.alphaField = NanoVdbSampledField::open(file, "alpha");
    }
    return result;
}

nlohmann::json writeNanoVdbMean(const MeanField& mean) {
    const auto& baked = dynamic_cast<const NanoVdbMean&>(mean);
    nlohmann::json field;
    field["mean_type"] = "nanovdb";
    field["grid_file"] = baked.gridFile().string();
    // The grid file is the source of truth for sigma -- a config may omit it
    // and the render then reads this value -- so recording it here says which
    // scale the field was read at, rather than feeding one back in.
    field["sigma"] = baked.sigma();
    field["use_alpha_grid"] = true;
    field["background"] = baked.background();
    field["voxel_size"] = baked.voxelSizeHint();
    if (const auto sidecar = readSidecar(baked.gridFile())) {
        field["coverage"] = sidecar->fullDomain ? "full_domain" : "surface_band";
    }
    field["grid_bounds"] = {{"min", {baked.gridBounds().minimum.x(), baked.gridBounds().minimum.y(),
                                     baked.gridBounds().minimum.z()}},
                            {"max", {baked.gridBounds().maximum.x(), baked.gridBounds().maximum.y(),
                                     baked.gridBounds().maximum.z()}}};
    return field;
}

} // namespace

void registerNanoVdbFieldTypes() {
    registerMeanFactory("nanovdb", makeNanoVdbMean);
    registerMeanWriter("nanovdb", writeNanoVdbMean);
}

} // namespace mf

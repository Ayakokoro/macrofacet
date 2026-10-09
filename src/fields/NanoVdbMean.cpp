#include "macrofacet/fields/NanoVdbMean.h"

#include "fieldgen/NanoVdbIO.h"
#include "fields/NanoVdbGridSampler.h"
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
    double cachedMaximumGradientNorm = 0.0;
    bool fullDomain = false;

    double maximumGradientNorm() const {
        return cachedMaximumGradientNorm;
    }
};

NanoVdbMean::NanoVdbMean() : impl_(std::make_unique<Impl>()) {}
NanoVdbMean::~NanoVdbMean() = default;

std::shared_ptr<const NanoVdbMean> NanoVdbMean::open(const std::filesystem::path& gridFile,
                                                     double sigmaOverride) {
    const auto sidecar = readSidecar(gridFile);
    if (sidecar && sidecar->version != 2) {
        throw std::invalid_argument(
            "NanoVDB field needs a version 2 sidecar for index-node samples; re-bake " +
            gridFile.string());
    }
    auto result = std::shared_ptr<NanoVdbMean>(new NanoVdbMean());
    result->impl_->path = gridFile;
    result->impl_->fullDomain = sidecar && sidecar->fullDomain;
    result->impl_->view = detail::GridView::open(gridFile, "sdf");

    // The primitive generator writes only sdf/density/alpha, without sigma
    // metadata. Its caller must provide sigma explicitly; our own bakes carry
    // a sigma grid and sidecar, which continue to be the source of truth.
    const auto storedSigma = readSigmaGrid(gridFile);
    const double fileSigma = (storedSigma || sidecar)
        ? resolveSigma(gridFile) : sigmaOverride;
    if (!(fileSigma > 0.0) || !std::isfinite(fileSigma)) {
        throw std::invalid_argument(
            "NanoVDB field has no stored sigma; provide field.sigma for " + gridFile.string());
    }
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
    result->impl_->cachedMaximumGradientNorm =
        result->impl_->view.maximumTrilinearGradientNorm();
    return result;
}

// Mesh bakes have a zero SDF background, but their separate density grid has a
// zero background too. Classic transport uses that density grid to skip vacuum.
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

MeanJet NanoVdbMean::evaluateInCell(const Point3& x, const Point3& interior) const {
    MeanJet jet;
    impl_->view.sampleWithGradientInCell(x, interior, jet.value, jet.gradient);
    if (!std::isfinite(jet.value) || !jet.gradient.allFinite())
        throw std::runtime_error("nonfinite NanoVDB ray-cell mean");
    return jet;
}

double NanoVdbMean::valueDifference(const Point3& x, const Vector3& displacement) const {
    return impl_->view.valueDifference(x, displacement);
}

MeanRayPoint NanoVdbMean::queryRayPoint(const Point3& origin, const Vector3& direction,
                                      double begin, double maximumEnd) const {
    if (!impl_->fullDomain)
        throw std::invalid_argument("point_linear requires NanoVDB coverage=full_domain");
    if (!origin.allFinite() || !direction.allFinite() || !(begin >= 0) ||
        !(direction.squaredNorm() > 0) || !std::isfinite(maximumEnd) || !(maximumEnd > begin))
        throw std::invalid_argument("invalid NanoVDB point-query interval");
    const double dx = impl_->view.voxelSize();
    const Point3 point = origin+begin*direction;
    double end = maximumEnd;
    for (int a = 0; a < 3; ++a) {
        if (direction[a] == 0) continue;
        double u = (point[a]-impl_->view.origin()[a])/dx;
        const double nearest = std::round(u);
        // Include the operands' scale: near index zero, origin + t*direction
        // cancels and abs(u) alone severely underestimates roundoff.
        const double coordinateScale = 1+(std::abs(origin[a])+std::abs(begin*direction[a])+
                                          std::abs(impl_->view.origin()[a]))/dx;
        if (std::abs(u-nearest) <= 32*std::numeric_limits<double>::epsilon()*coordinateScale)
            u = nearest;
        const double face = direction[a] > 0 ? std::floor(u)+1 : std::ceil(u)-1;
        const double t = (impl_->view.origin()[a]+face*dx-origin[a])/direction[a];
        if (t > begin) end = std::min(end,t);
    }
    MeanRayPoint result;
    result.end = end;
    const Point3 interior = origin+(begin+0.5*(end-begin))*direction;
    // Validate and interpolate the same eight corners in one current-point query.
    impl_->view.sampleWithGradientInCell(point,interior,result.jet.value,result.jet.gradient,true);
    if (!std::isfinite(result.jet.value) || !result.jet.gradient.allFinite())
        throw std::runtime_error("nonfinite NanoVDB point query");
    return result;
}

MeanRayBounds NanoVdbMean::rayBounds(const Point3& origin, const Vector3& direction,
                                     double begin, double end) const {
    std::vector<double> knots{begin};
    appendRayBreakpoints(origin, direction, begin, end, knots);
    knots.push_back(end);
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    MeanRayBounds result;
    result.minimumValue = result.minimumDerivative = std::numeric_limits<double>::infinity();
    result.maximumValue = result.maximumDerivative = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < knots.size(); ++i) {
        const MeanRayBounds piece = impl_->view.rayCellBounds(origin, direction, knots[i-1], knots[i]);
        result.minimumValue = std::min(result.minimumValue, piece.minimumValue);
        result.maximumValue = std::max(result.maximumValue, piece.maximumValue);
        result.minimumDerivative = std::min(result.minimumDerivative, piece.minimumDerivative);
        result.maximumDerivative = std::max(result.maximumDerivative, piece.maximumDerivative);
        if (i == 1) {
            result.beginDerivative = piece.beginDerivative;
            if (knots.size() == 2) result.maximumSecondDerivative = piece.maximumSecondDerivative;
        }
    }
    result.certified = knots.size() >= 2;
    return result;
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
        const double u0 = (origin[axis] + begin * speed - gridOrigin[axis]) / dx;
        const double u1 = (origin[axis] + end * speed - gridOrigin[axis]) / dx;
        const long long first = static_cast<long long>(std::floor(std::min(u0, u1))) + 1;
        const long long last = static_cast<long long>(std::ceil(std::max(u0, u1))) - 1;
        if (last - first > 100000) {
            throw std::invalid_argument("ray crosses too many NVDB interpolation cells");
        }
        for (long long index = first; index <= last; ++index) {
            const double coordinate = gridOrigin[axis] + static_cast<double>(index) * dx;
            const double age = (coordinate - origin[axis]) / speed;
            if (age > begin && age < end) knots.push_back(age);
        }
    }
}

double NanoVdbMean::sigma() const { return impl_->sigma; }

void NanoVdbMean::requireFullRayCoverage(const Point3& origin, const Vector3& direction,
                                        double maximumDistance) const {
    if (!impl_->fullDomain) {
        throw std::invalid_argument("GP first passage requires NanoVDB coverage=full_domain "
                                    "in its sidecar: " + impl_->path.string());
    }
    if (!origin.allFinite() || !direction.allFinite() ||
        !(direction.squaredNorm() > 0.0) || !std::isfinite(maximumDistance) ||
        !(maximumDistance > 0.0)) {
        throw std::invalid_argument("invalid NanoVDB first-passage ray");
    }
    const auto covered = [&](double t) {
        if (!impl_->view.hasCompleteInterpolationCell(origin + t * direction)) {
            throw std::invalid_argument("NanoVDB first-passage ray reaches missing/non-finite "
                "SDF interpolation nodes at distance " + std::to_string(t) +
                "; shorten the ray or re-bake a larger full domain: " + impl_->path.string());
        }
    };
    std::vector<double> knots{0.0, maximumDistance};
    appendRayBreakpoints(origin, direction, 0.0, maximumDistance, knots);
    std::sort(knots.begin(), knots.end());
    // Every open interval is one interpolation cell. Its eight stored corners
    // cover its CLOSED endpoints too. Do not select an unrelated outside cell
    // by flooring an endpoint on the outermost stored node / upper grid face.
    for (std::size_t i = 1; i < knots.size(); ++i) {
        if (knots[i] > knots[i - 1]) covered(0.5 * (knots[i] + knots[i - 1]));
    }
}

// Exactly sigma(), offered through the interface a config can consult without
// knowing what a NanoVdbMean is.
std::optional<double> NanoVdbMean::intrinsicSigma() const { return sigma(); }

double NanoVdbMean::background() const { return impl_->view.background(); }
const std::filesystem::path& NanoVdbMean::gridFile() const { return impl_->path; }
const Bounds3& NanoVdbMean::gridBounds() const { return impl_->view.worldBounds(); }
const Bounds3& NanoVdbMean::activeNodeBounds() const { return impl_->view.activeNodeBounds(); }

namespace {

MeanBuildResult makeNanoVdbMean(const nlohmann::json& fieldJson) {
    const std::string file = fieldJson.at("grid_file").get<std::string>();
    const double sigmaOverride = fieldJson.value("sigma", 0.0);
    MeanBuildResult result;
    result.mean = NanoVdbMean::open(file, sigmaOverride);
    const auto& baked = static_cast<const NanoVdbMean&>(*result.mean);
    result.activeDomain = baked.activeNodeBounds();
    if (!result.activeDomain->valid()) {
        throw std::invalid_argument("nanovdb SDF active nodes must span at least two samples per axis");
    }
    // The alpha grid lives in the same file, so a config only ever names one
    // path. It is optional: the generator always writes it, but a file trimmed
    // down to the sdf grid stays usable (the material then falls back to the
    // config's roughness).
    if (fieldJson.value("use_alpha_grid", true) && nanovdb::io::hasGrid(file, "alpha")) {
        result.alphaField = NanoVdbSampledField::open(file, "alpha");
        const auto& alpha = static_cast<const NanoVdbSampledField&>(*result.alphaField);
        if (alpha.background() == 0.0) {
            const auto density = NanoVdbSampledField::open(file, "density");
            if (!alpha.positiveAtEveryPositiveNodeOf(*density)) {
                throw std::invalid_argument(
                    "zero-background alpha must be positive wherever baked density is positive");
            }
            result.alphaPositiveOnDensitySupport = true;
        }
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

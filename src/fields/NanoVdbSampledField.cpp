#include "macrofacet/fields/NanoVdbSampledField.h"

#include "fields/NanoVdbGridSampler.h"

#include <stdexcept>
#include <utility>

namespace mf {

struct NanoVdbSampledField::Impl {
    detail::GridView view;
    std::string name;
};

NanoVdbSampledField::NanoVdbSampledField() : impl_(std::make_unique<Impl>()) {}
NanoVdbSampledField::~NanoVdbSampledField() = default;

std::shared_ptr<const NanoVdbSampledField> NanoVdbSampledField::open(
    const std::filesystem::path& gridFile, const std::string& gridName) {
    auto result = std::shared_ptr<NanoVdbSampledField>(new NanoVdbSampledField());
    result->impl_->name = gridName;
    result->impl_->view = detail::GridView::open(gridFile, gridName);
    return result;
}

double NanoVdbSampledField::sample(const Point3& x) const {
    const double value = impl_->view.sample(x);
    return std::isfinite(value) ? value : impl_->view.background();
}

ScalarBounds NanoVdbSampledField::bounds(const Bounds3& domain) const {
    const auto range = impl_->view.bounds(domain);
    ScalarBounds result;
    result.minimumValue = range.first;
    result.maximumValue = range.second;
    result.certified = true;
    return result;
}

double NanoVdbSampledField::minimumValue() const { return impl_->view.minimumValue(); }
double NanoVdbSampledField::maximumValue() const { return impl_->view.maximumValue(); }
double NanoVdbSampledField::background() const { return impl_->view.background(); }
const std::string& NanoVdbSampledField::gridName() const { return impl_->name; }
const Bounds3& NanoVdbSampledField::gridBounds() const { return impl_->view.worldBounds(); }
double NanoVdbSampledField::voxelSize() const { return impl_->view.voxelSize(); }

} // namespace mf

#include "macrofacet/fields/MeanFactory.h"
#include "macrofacet/gpss/ShaderBallMean.h"
#include <nlohmann/json.hpp>
#include <map>
#include <stdexcept>
#include <utility>

namespace mf {
namespace {

Vector3 vector3(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) throw std::invalid_argument("expected a 3-vector");
    return Vector3(value[0].get<double>(), value[1].get<double>(), value[2].get<double>());
}

std::map<std::string, MeanFactoryFn>& registry() {
    // Construct-on-first-use: safe regardless of static initialization order,
    // which matters because macrofacet_field registers into it from its own
    // translation units' initialization.
    static std::map<std::string, MeanFactoryFn> table;
    return table;
}

std::map<std::string, MeanWriterFn>& writerRegistry() {
    static std::map<std::string, MeanWriterFn> table;
    return table;
}

MeanBuildResult onlyMean(MeanFieldPtr mean) {
    return {std::move(mean), nullptr};
}

void registerBuiltins() {
    registerMeanFactory("plane", [](const nlohmann::json& fieldJson) {
        return onlyMean(std::make_shared<PlaneMean>(vector3(fieldJson.at("plane_normal")),
                                                    fieldJson.at("plane_offset").get<double>()));
    });
    registerMeanFactory("sphere", [](const nlohmann::json& fieldJson) {
        return onlyMean(std::make_shared<SphereMean>(vector3(fieldJson.at("sphere_center")),
                                                     fieldJson.at("sphere_radius").get<double>()));
    });
    registerMeanFactory("cutaway_sphere", [](const nlohmann::json& fieldJson) {
        return onlyMean(std::make_shared<CutawaySphereMean>(
            vector3(fieldJson.at("sphere_center")), fieldJson.at("inner_radius").get<double>(),
            fieldJson.at("outer_radius").get<double>()));
    });
    registerMeanFactory("shader_ball", [](const nlohmann::json& fieldJson) {
        return onlyMean(std::make_shared<ShaderBallMean>(
            vector3(fieldJson.at("sphere_center")), fieldJson.at("sphere_radius").get<double>(),
            vector3(fieldJson.at("groove_axis")), fieldJson.at("groove_radius").get<double>()));
    });
    registerMeanFactory("constant", [](const nlohmann::json& fieldJson) {
        return onlyMean(std::make_shared<ConstantMean>(
            fieldJson.at("constant_value").get<double>()));
    });
}

struct BuiltinRegistrar {
    BuiltinRegistrar() { registerBuiltins(); }
};
const BuiltinRegistrar kBuiltinRegistrar;

} // namespace

void registerMeanFactory(const std::string& type, MeanFactoryFn fn) {
    if (type.empty()) throw std::invalid_argument("mean type must not be empty");
    registry()[type] = std::move(fn);
}

void registerMeanWriter(const std::string& type, MeanWriterFn fn) {
    if (type.empty()) throw std::invalid_argument("mean type must not be empty");
    writerRegistry()[type] = std::move(fn);
}

std::optional<nlohmann::json> writeMeanToJson(const MeanField& mean) {
    const auto it = writerRegistry().find(mean.typeName());
    if (it == writerRegistry().end()) return std::nullopt;
    return it->second(mean);
}

MeanBuildResult buildMeanFromJson(const nlohmann::json& fieldJson) {
    const std::string type = fieldJson.at("mean_type").get<std::string>();
    const auto it = registry().find(type);
    if (it == registry().end()) {
        throw std::invalid_argument("unknown mean type: " + type);
    }
    return it->second(fieldJson);
}

} // namespace mf

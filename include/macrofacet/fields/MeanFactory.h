#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/gpss/MeanField.h"
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

namespace mf {

// What a mean type builds. `alphaField` is the per-point material NDF field and
// is null for every procedural type. It travels with the mean because both come
// from the same baked file, which the config names exactly once -- but they stay
// two independent fields, because they feed two different roles in the model
// (see docs/archive/PLAN_NANOVDB_FIELD.md 1.3).
struct MeanBuildResult {
    MeanFieldPtr mean;
    ScalarFieldPtr alphaField;
    // Baked fields may define their own tracing domain. Procedural fields leave
    // this empty and require domain_min/domain_max in the config.
    std::optional<Bounds3> activeDomain;
};

// Registry of mean-field builders keyed by the config's "mean_type".
//
// This exists so the core library never has to know about NanoVDB: it owns the
// procedural types (plane / sphere / cutaway_sphere / shader_ball / constant),
// and macrofacet_field adds "nanovdb" at startup by calling
// registerMeanFactory(). Nothing here depends on the field library.
using MeanFactoryFn = std::function<MeanBuildResult(const nlohmann::json& fieldJson)>;

void registerMeanFactory(const std::string& type, MeanFactoryFn fn);

// Throws std::invalid_argument naming the offending type when it is unknown.
MeanBuildResult buildMeanFromJson(const nlohmann::json& fieldJson);

// Echoes a built field back into the config schema, so a resolved config can be
// fed straight back in. Same registry, same keys: a type that can be built
// should be able to describe itself.
using MeanWriterFn = std::function<nlohmann::json(const MeanField&)>;

void registerMeanWriter(const std::string& type, MeanWriterFn fn);

// std::nullopt when the field's typeName() has no registered writer, which is
// not an error -- the caller decides whether to omit the field from the output.
std::optional<nlohmann::json> writeMeanToJson(const MeanField& mean);

} // namespace mf

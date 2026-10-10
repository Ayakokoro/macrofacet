#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

void testTransmittanceConfig(TestContext& context) {
    using namespace mf;
    namespace fs = std::filesystem;
    const fs::path directory = fs::temp_directory_path();
    const fs::path input = directory / "macrofacet_transmittance_config_test.json";
    const fs::path resolved = directory / "macrofacet_transmittance_resolved_test.json";
    nlohmann::json root = {
        {"schema_version", 1},
        {"seed", 77},
        {"field", {
            {"mean_type", "plane"}, {"plane_normal", {0.0, 0.0, 1.0}},
            {"plane_offset", 0.0}, {"sigma", 0.1},
            {"correlation_lengths", {0.2, 0.2, 0.2}},
            {"domain_min", {-1.0, -1.0, -1.0}},
            {"domain_max", {1.0, 1.0, 1.0}}}},
        {"material", {
            {"ndf_family", "generalized_gaussian"},
            {"eta_rgb", {0.2, 0.9, 1.1}}, {"k_rgb", {3.9, 2.5, 2.2}}}},
        {"transport", {{"roulette_start_depth", 4}}},
        {"numeric", {
            {"relative_tolerance", 1e-6}, {"absolute_tolerance", 1e-9},
            {"max_quadrature_subdivisions", 1024}, {"max_root_iterations", 64}}},
        {"render", {
            {"width", 4}, {"height", 4}, {"samples_per_pixel", 1},
            {"camera_position", {0.0, 0.0, 1.0}},
            {"camera_target", {0.0, 0.0, 0.0}},
            {"vertical_fov_degrees", 45.0}, {"environment", "white"}}},
        {"transmittance", {
            {"modes", {"classic_local", "classic_global", "global_conditional"}},
            {"curve", {{"bins", 32}, {"spacing", "linear"},
                       {"distance_origin", "ray_origin"}}},
            {"monte_carlo", {{"trials_per_ray", 123}, {"confidence_level", 0.9},
                             {"write_raw_samples", true}}},
            {"conditional_birth", {{"policy", "sample_positive_exterior"}}},
            {"comparison", {
                {"local_transport_reference", "classic_global"},
                {"constant_exponential", {{"enabled", true},
                                           {"fit", "censored_mle"}}}}},
            {"rays", {
                {{"id", "outside"}, {"group", "batch"},
                 {"origin", {0.0, 0.0, 2.0}}, {"direction", {0.0, 0.0, -2.0}},
                 {"max_distance", 3.0}},
                {{"id", "fixed"}, {"origin", {0.0, 0.0, 0.0}},
                 {"direction", {1.0, 0.0, 0.1}}, {"max_distance", 0.5},
                 {"bins", 17},
                 {"conditional_birth", {{"policy", "fixed_observation"},
                                          {"value", 0.0},
                                          {"gradient", {0.0, 0.0, 1.0}}}}}}}}},
        {"output_directory", "outputs/transmittance_config_test"}};
    {
        std::ofstream stream(input);
        stream << root;
    }
    const ExperimentConfig config = loadExperimentConfig(input);
    context.require(config.transmittance.has_value(),
                    "transmittance block is parsed");
    if (config.transmittance) {
        const TransmittanceConfig& transmittance = *config.transmittance;
        context.require(transmittance.bins == 32 && transmittance.trialsPerRay == 123,
                        "transmittance budgets are parsed");
        context.near(transmittance.confidenceLevel, 0.9, 0.0,
                     "transmittance confidence is parsed");
        context.require(transmittance.rays.size() == 2,
                        "explicit transmittance rays are parsed");
        context.near(transmittance.rays[0].direction.norm(), 1.0, 1e-15,
                     "ray directions are normalized while loading");
        context.require(transmittance.rays[1].bins == 17 &&
                        transmittance.rays[1].conditionalBirth &&
                        transmittance.rays[1].conditionalBirth->policy == "fixed_observation",
                        "per-ray curve and conditional birth overrides are parsed");
    }
    writeResolvedConfig(config, resolved);
    nlohmann::json written;
    {
        std::ifstream stream(resolved);
        stream >> written;
    }
    context.require(written.at("transmittance").at("rays").size() == 2 &&
                    written.at("transmittance").at("monte_carlo")
                        .at("trials_per_ray").get<int>() == 123,
                    "resolved config preserves the transmittance experiment");
    root["transport"]["renewal"] = {{"model","unused.json"},{"profile_mode","point_linear"}};
    { std::ofstream stream(input); stream << root; }
    const auto pointConfig = loadExperimentConfig(input);
    context.require(pointConfig.renewal.profileMode == "point_linear","point profile config parsed");
    writeResolvedConfig(pointConfig,resolved);
    { std::ifstream stream(resolved); stream >> written; }
    context.require(written["transport"]["renewal"]["profile_mode"] == "point_linear",
                    "resolved config records the approximation explicitly");
    for (const auto& [key,value] : std::vector<std::pair<std::string,int>>{
            {"ray_pool_size",16384},{"batch_size",8192},{"auxiliary_batch_minimum",256},{"maximum_queue_delay",4},{"max_in_flight_batches",3}})
        root["transport"]["renewal"][key] = value;
    { std::ofstream stream(input); stream << root; }
    const auto queueConfig = loadExperimentConfig(input);
    writeResolvedConfig(queueConfig,resolved);
    { std::ifstream stream(resolved); stream >> written; }
    context.require(queueConfig.renewal.rayPoolSize == 16384 && queueConfig.renewal.batchSize == 8192 &&
        written["transport"]["renewal"]["ray_pool_size"] == 16384 &&
        written["transport"]["renewal"]["auxiliary_batch_minimum"] == 256 &&
        written["transport"]["renewal"]["maximum_queue_delay"] == 4 &&
        queueConfig.renewal.maximumInFlightBatches == 3 &&
        written["transport"]["renewal"]["max_in_flight_batches"] == 3,
        "queue scheduling configuration roundtrips");
    for (const auto& [key,value] : std::vector<std::pair<std::string,int>>{
            {"ray_pool_size",-1},{"ray_pool_size",65537},
            {"auxiliary_batch_minimum",0},{"maximum_queue_delay",-1},{"max_in_flight_batches",0},{"max_in_flight_batches",5}}) {
        auto invalid = root; invalid["transport"]["renewal"][key] = value;
        { std::ofstream stream(input); stream << invalid; }
        bool rejected = false;
        try { (void)loadExperimentConfig(input); } catch (const std::invalid_argument&) { rejected = true; }
        context.require(rejected,"invalid queue configuration rejected");
    }
    root["transport"]["renewal"]["profile_mode"] = "unknown";
    { std::ofstream stream(input); stream << root; }
    bool rejectedProfile = false;
    try { (void)loadExperimentConfig(input); } catch (const std::invalid_argument&) { rejectedProfile = true; }
    context.require(rejectedProfile,"unknown mean profile mode rejected");
    root["transport"].erase("renewal");
    root["transport"]["first_passage_model"] = {{"type", "mlp"}};
    { std::ofstream stream(input); stream << root; }
    bool rejectedRemovedModel=false;
    try { (void)loadExperimentConfig(input); }
    catch (const std::invalid_argument&) { rejectedRemovedModel=true; }
    context.require(rejectedRemovedModel, "removed learned-model configuration is rejected explicitly");
    fs::remove(input);
    fs::remove(resolved);
}

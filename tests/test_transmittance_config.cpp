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
    auto hybridRoot = root;
    hybridRoot.erase("transmittance");
    hybridRoot["transport"] = {
        {"mode","hybrid_first_hit"},
        {"first_hit",{{"method","gpis_function_space"},{"backend","scalar"}}},
        {"continuation",{{"method","global_conditional"},{"backend","scalar"}}}};
    { std::ofstream stream(input); stream << hybridRoot; }
    auto hybrid = loadExperimentConfig(input);
    prepareRenewalModel(hybrid);
    prepareFirstHitModel(hybrid);
    context.require(hybrid.continuationMethod == "global_conditional" &&
        hybrid.material.gpModel == GpModel::GlobalPointwise && !hybrid.renewal.model && !hybrid.firstHit.model,
        "GPIS plus conditional config loads and prepares without neural checkpoints");
    auto renewalRoot=root;
    renewalRoot["transport"]["mode"]="global_conditional_renewal_plus";
    renewalRoot["transport"]["global_conditional_renewal_plus"]={{"renewal_distance",0.0125}};
    renewalRoot["transmittance"]["modes"]={"global_conditional","global_conditional_renewal_plus"};
    { std::ofstream stream(input); stream << renewalRoot; }
    auto conditionalRenewal=loadExperimentConfig(input);
    prepareRenewalModel(conditionalRenewal);
    context.near(conditionalRenewal.conditionalRenewalDistance,0.0125,0,"standalone conditional renewal distance is parsed");
    context.require(conditionalRenewal.material.gpModel==GpModel::GlobalPointwise && !conditionalRenewal.renewal.model,
        "conditional renewal+ selects the global field without loading a checkpoint");
    context.near((conditionalRenewal.field.kernel.precision()-config.field.kernel.precision()).norm(),0,0,
        "conditional renewal+ preserves the configured global kernel");
    auto renewalHybridRoot=hybridRoot;
    renewalHybridRoot["transport"]["global_conditional_renewal_plus"]={{"renewal_distance",0.0125}};
    renewalHybridRoot["transport"]["continuation"]={{"method","global_conditional_renewal_plus"},{"renewal_distance",0.02}};
    { std::ofstream stream(input); stream << renewalHybridRoot; }
    auto renewalHybrid=loadExperimentConfig(input);
    prepareRenewalModel(renewalHybrid); prepareFirstHitModel(renewalHybrid);
    writeResolvedConfig(renewalHybrid,resolved);
    { std::ifstream stream(resolved); stream >> written; }
    context.require(written["transport"]["continuation"]["renewal_distance"]==0.02 &&
        written["transport"]["continuation"]["distance_units"]=="world" &&
        written["transport"]["continuation"]["boundary_policy"]=="sample_previous_segment_positive_endpoint" &&
        written["transport"]["continuation"]["kernel_policy"]=="shared_global_kernel" &&
        written["transport"]["continuation"]["mean_policy"]=="global_field" &&
        written["transport"]["continuation"]["gp_model"]=="global_pointwise" &&
        !written["transport"].contains("local_conditional") &&
        !written["transport"].contains("renewal"),"resolved renewal+ continuation records its global field and endpoint policy");
    for (double step : {0.0,-0.01}) {
        renewalHybridRoot["transport"]["continuation"]["renewal_distance"]=step;
        { std::ofstream stream(input); stream << renewalHybridRoot; }
        bool rejected=false;
        try { (void)loadExperimentConfig(input); } catch(const std::invalid_argument&) { rejected=true; }
        context.require(rejected,"nonpositive conditional renewal distance is rejected");
    }
    const auto rejectConfig = [&](const nlohmann::json& invalid, const std::string& message) {
        { std::ofstream stream(input); stream << invalid; }
        bool rejected=false;
        try { (void)loadExperimentConfig(input); } catch(const std::invalid_argument&) { rejected=true; }
        context.require(rejected,message);
    };
    auto removed=root;
    removed["transport"]["mode"]="local_conditional";
    rejectConfig(removed,"removed local_conditional mode is not an alias");
    removed=root;
    removed["transport"]["local_conditional"]={{"step_size",0.01}};
    rejectConfig(removed,"removed local_conditional settings are not silently ignored");
    removed=hybridRoot;
    removed["transport"]["continuation"]["method"]="local_conditional";
    rejectConfig(removed,"removed local_conditional continuation is rejected");
    removed=root;
    removed["transmittance"]["modes"]={"local_conditional"};
    rejectConfig(removed,"removed local_conditional transmittance mode is rejected");
    removed=renewalRoot;
    removed["transport"]["global_conditional_renewal_plus"]={{"step_size",0.01}};
    rejectConfig(removed,"renewal+ rejects the obsolete step_size parameter");
    for (double distance : {0.0,-0.01}) {
        auto invalid=renewalRoot;
        invalid["transport"]["global_conditional_renewal_plus"]["renewal_distance"]=distance;
        rejectConfig(invalid,"standalone renewal distance must be positive");
    }
    auto invalidMaterial=conditionalRenewal;
    invalidMaterial.material.ndfFamily=NdfFamily::GGXBaseline;
    bool rejectedMaterial=false;
    try { prepareRenewalModel(invalidMaterial); } catch(const std::invalid_argument&) { rejectedMaterial=true; }
    context.require(rejectedMaterial,"renewal+ requires the same Gaussian material as global conditional");
    hybridRoot["transport"]["first_hit"]={{"method",kTungstenFunctionSpaceMethod},
        {"sampler",{{"sample_points",8},{"step_size",.01}}}};
    hybridRoot["transport"]["continuation"]={{"method",kTungstenFunctionSpaceMethod},
        {"sampler",{{"sample_points",12},{"step_size",.02}}}};
    { std::ofstream stream(input); stream << hybridRoot; }
    auto tungsten=loadExperimentConfig(input);
    prepareRenewalModel(tungsten); prepareFirstHitModel(tungsten);
    context.require(!tungsten.firstHit.model && !tungsten.renewal.model &&
        tungsten.firstHit.tungsten.samplePoints==8 && tungsten.continuationTungsten.samplePoints==12,
        "Tungsten first-hit and continuation have independent configs without checkpoints");
    context.near(tungsten.firstHit.tungsten.stepSize,.01,0,"Tungsten config retains world step");
    writeResolvedConfig(tungsten,resolved);
    { std::ifstream stream(resolved); stream >> written; }
    context.require(written["transport"]["first_hit"]["step_units"]=="world" &&
        written["transport"]["continuation"]["sampler"]["sample_points"]==12 &&
        !written["transport"].contains("renewal"),"resolved Tungsten config identifies units and both samplers");
    for (const auto& bad : {nlohmann::json{{"history_points",1}},
                            nlohmann::json{{"correlation_context","global"}},
                            nlohmann::json{{"skip_space",.01}},
                            nlohmann::json{{"sample_points",1}}}) {
        auto invalid=hybridRoot;
        invalid["transport"]["continuation"]["sampler"]=bad;
        { std::ofstream stream(input); stream << invalid; }
        bool rejected=false;
        try { (void)loadExperimentConfig(input); } catch(const std::invalid_argument&) { rejected=true; }
        context.require(rejected,"unsupported Tungsten semantics fail explicitly");
    }
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

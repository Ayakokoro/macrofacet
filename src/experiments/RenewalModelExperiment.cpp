#include "macrofacet/experiments/RenewalModelExperiment.h"
#include "macrofacet/transport/RenewalRayDistribution.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace mf {
void runRenewalModelExperiment(const FirstPassageExperimentConfig& config,
    const std::filesystem::path& bundle, const std::filesystem::path& source) {
    using Json = nlohmann::json;
    if (config.initialConditionType == "fixed_value" || config.fixedEndpoint.enabled)
        throw std::invalid_argument("renewal-query requires an A/B first-passage configuration without fixed endpoints");
    const auto model = RenewalHazardModel::load(bundle);
    for (const auto& kernel : config.kernels)
        if (kernel.type != covarianceKernelTypeName(model.kernelType()))
            throw std::invalid_argument("renewal-query reference/model kernel mismatch");
    std::ifstream input(source);
    Json root;
    input >> root;
    const auto options = root.value("renewal_query", Json::object());
    std::vector<double> xs;
    if (options.contains("normalized_distances")) xs = options.at("normalized_distances").get<std::vector<double>>();
    else for (int i = 0; i <= config.curveBins; ++i) xs.push_back(config.maximumTime*i/config.curveBins);
    for (double x : xs) if (!(x >= 0 && x <= config.maximumTime))
        throw std::invalid_argument("renewal_query.normalized_distances must lie within the profile");
    std::vector<std::array<double, 2>> intervals{{0, config.maximumTime}};
    if (options.contains("survival_intervals"))
        intervals = options.at("survival_intervals").get<std::vector<std::array<double, 2>>>();
    const auto depths = options.value("optical_depths", std::vector<double>{0.01, 0.1, 0.5, 1.0, 3.0, 10.0});
    const int speedTrials = options.value("speed_sample_count", 0);
    if (speedTrials < 0) throw std::invalid_argument("negative speed sample count");
    const std::vector<double> speedThresholds{1e-4, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30};
    Json result = {{"schema_version", 1}, {"checkpoint_sha256", model.checkpointSha256()},
                   {"kernel_type", covarianceKernelTypeName(model.kernelType())},
                   {"model", bundle.string()}, {"seed", config.seed}, {"rays", Json::array()}};
    std::size_t rayIndex = 0;
    for (const auto& kernel : config.kernels) for (const auto& condition : config.collisionStates) {
        const double sigma = std::sqrt(kernel.variance), ell = kernel.lengthScale;
        RayMeanProfile mean = [&]() {
            if (!config.processMeanField)
                return RayMeanProfile::affine(condition.beta0, condition.betaMeanSlope,
                                             config.maximumTime, config.profileMaximumStep);
            const auto& ray = condition.ray.value();
            return RayMeanProfile::fromField(*config.processMeanField, ray.origin, ray.direction,
                sigma, ell, config.maximumTime*ell, config.profileMaximumStep);
        }();
        const bool surface = config.initialConditionType == "collision_state";
        const double g = !surface ? 0 : config.processMeanField
            ? ell/sigma * condition.ray->gradient.dot(condition.ray->direction) : condition.betaCollisionSlope;
        const RayStartCondition start{surface ? RayStartMode::SurfaceOutward : RayStartMode::PositiveExterior, g};
        const auto distribution = RenewalRayDistribution::fromModel(model, mean, start, ell);
        std::vector<RenewalHazardModel::State> entering;
        if (model.hasMixture()) {
            auto state = model.initialize(mean.segments().front(), start);
            for (const auto& segment : mean.segments()) {
                entering.push_back(state);
                state = model.evaluate(state, segment).nextState;
            }
        }
        const double b0 = mean.segments().front().value(0);
        Json item = {{"kernel_id", kernel.id}, {"state_id", condition.id}, {"start_mode", config.initialConditionType},
            {"sigma", sigma}, {"ell", ell}, {"maximum_x", mean.maximumX()},
            {"known_initial", {surface ? 1.0 : 0.0, b0, surface ? -b0 : 0.0,
                surface ? g-mean.segments().front().derivative(0) : 0.0}},
            {"segments", Json::array()}, {"queries", Json::array()}, {"intervals", Json::array()}};
        for (std::size_t i = 0; i < mean.segments().size(); ++i) {
            const auto& segment = mean.segments()[i];
            item["segments"].push_back({{"begin", segment.begin}, {"end", segment.end},
                {"features", segment.features()}, {"rates", distribution.segments()[i].rates()}});
        }
        Random rng(config.seed + 0x9e3779b97f4a7c15ULL*(++rayIndex));
        std::vector<double> hits;
        for (int i = 0; i < config.trajectories; ++i) {
            const auto sample = distribution.sample(rng);
            if (sample.hit) hits.push_back(sample.distance);
        }
        std::sort(hits.begin(), hits.end());
        item["sample_count"] = config.trajectories;
        item["sampled_hits"] = hits.size();
        for (double x : xs) {
            const double distance = x*ell;
            const auto count = std::upper_bound(hits.begin(), hits.end(), distance)-hits.begin();
            item["queries"].push_back({{"x", x}, {"distance", distance},
                {"cumulative_hazard", distribution.cumulativeHazard(distance)},
                {"transmittance", distribution.transmittance(distance)},
                {"physical_hazard", distribution.hazard(distance)},
                {"sampled_survival", 1.0-static_cast<double>(count)/config.trajectories}});
            if (model.hasMixture()) {
                std::size_t index = 0;
                while (index+1 < mean.segments().size() && x >= mean.segments()[index].end) ++index;
                const auto& segment = mean.segments()[index];
                const double u = std::clamp((x-segment.begin)/(segment.end-segment.begin), 0.0, 1.0);
                const auto mixture = model.mixture(entering[index], segment, u);
                auto& query = item["queries"].back();
                query["speed_mixture"] = {{"weights", mixture.weights}, {"means", mixture.means}, {"scales", mixture.scales}};
                query["speed_sample_count"] = speedTrials;
                if (speedTrials) {
                    std::vector<int> counts(speedThresholds.size(), 0);
                    for (int trial = 0; trial < speedTrials; ++trial) {
                        const double w = mixture.sample(rng);
                        if (!(w > 0) || !std::isfinite(w)) throw std::runtime_error("invalid sampled Renewal speed");
                        for (std::size_t j = 0; j < counts.size(); ++j) counts[j] += w <= speedThresholds[j];
                    }
                    query["speed_cdf"] = Json::array();
                    for (std::size_t j = 0; j < counts.size(); ++j)
                        query["speed_cdf"].push_back({{"w", speedThresholds[j]}, {"empirical", counts[j]/static_cast<double>(speedTrials)}});
                }
            }
        }
        for (const auto& bounds : intervals) {
            const double from = bounds[0]*ell, to = bounds[1]*ell;
            Json interval = {{"begin_x", bounds[0]}, {"end_x", bounds[1]},
                {"transmittance", distribution.transmittance(from, to)}, {"inversions", Json::array()}};
            for (double depth : depths) {
                const auto sample = distribution.sampleOpticalDepth(depth, from, to, 1e-10*ell);
                interval["inversions"].push_back({{"optical_depth", depth}, {"hit", sample.hit},
                    {"distance", sample.distance}, {"x", sample.distance/ell}, {"segment", sample.segment}});
            }
            item["intervals"].push_back(std::move(interval));
        }
        result["rays"].push_back(std::move(item));
    }
    std::filesystem::create_directories(config.outputDirectory);
    std::ofstream output(config.outputDirectory/"renewal_query.json");
    if (!output) throw std::runtime_error("cannot write Renewal query results");
    output << std::setw(2) << result << '\n';
    writeResolvedFirstPassageConfig(config, config.outputDirectory/"resolved_renewal_config.json");
}
} // namespace mf

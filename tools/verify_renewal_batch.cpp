// Replay renewal-query ray profiles through batched LibTorch inference.
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/transport/RenewalRayDistribution.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: macrofacet_verify_renewal_batch <model.json> <renewal_query.json> <backend> <report.json>\n";
        return 2;
    }
    try {
        using namespace mf;
        using Json = nlohmann::json;
        const auto model = RenewalHazardModel::load(argv[1]);
        std::ifstream source(argv[2]);
        Json reference; source >> reference;
        if (reference.at("checkpoint_sha256") != model.checkpointSha256())
            throw std::invalid_argument("batch reference and model checkpoints differ");
        const auto& rays = reference.at("rays");
        const int n = static_cast<int>(rays.size());
        RenewalBatchSession batch(model,n,argv[3]);
        std::vector<std::vector<RayMeanSegment>> profiles(n);
        std::vector<std::vector<RenewalHazardSegment>> hazards(n);
        std::vector<int> ids(n); std::iota(ids.begin(),ids.end(),0);
        std::vector<RayMeanSegment> first;
        std::vector<RayStartCondition> starts;
        std::size_t maximumSegments = 0, maximumQueries = 0, segmentCount = 0;
        for (int i = 0; i < n; ++i) {
            for (const auto& segment : rays[i].at("segments")) {
                const auto f = segment.at("features").get<std::array<double,5>>();
                const double delta = f[1]-f[0];
                profiles[i].push_back({segment.at("begin"),segment.at("end"),
                    {-2*delta+f[2]+f[3],3*delta-2*f[2]-f[3],f[2],f[0]}});
            }
            first.push_back(profiles[i].front());
            const auto known = rays[i].at("known_initial").get<std::array<double,4>>();
            starts.push_back(known[0] == 0 ? RayStartCondition{} :
                RayStartCondition{RayStartMode::SurfaceOutward,known[3]+first.back().derivative(0)});
            maximumSegments = std::max(maximumSegments,profiles[i].size());
            maximumQueries = std::max(maximumQueries,rays[i].at("queries").size());
        }
        batch.initialize(ids,first,starts);
        double maxRate = 0, maxMixture = 0, maxCumulative = 0, maxTransmittance = 0;
        const auto compare = [](double actual, double expected, double& maximum, const char* label) {
            const double error = std::abs(actual-expected);
            maximum = std::max(maximum,error);
            if (!std::isfinite(actual) || error > 4e-6+5e-5*std::abs(expected))
                throw std::runtime_error(std::string("batch parity failed: ")+label);
        };
        for (std::size_t step = 0; step < maximumSegments; ++step) {
            ids.clear(); std::vector<RayMeanSegment> segments;
            for (int i = n-1; i >= 0; --i) if (step < profiles[i].size()) {
                ids.push_back(i); segments.push_back(profiles[i][step]);
            }
            const auto rates = batch.evaluate(ids,segments);
            segmentCount += ids.size();
            for (std::size_t j = 0; j < ids.size(); ++j) {
                const auto expected = rays[ids[j]]["segments"][step]["rates"].get<std::array<double,4>>();
                for (int k = 0; k < 4; ++k) compare(rates[j][k],expected[k],maxRate,"hazard");
                hazards[ids[j]].emplace_back(segments[j].begin,segments[j].end,rates[j]);
            }
            for (std::size_t q = 0; q < maximumQueries; ++q) {
                std::vector<int> queryIds; std::vector<double> coordinates;
                for (int id : ids) {
                    if (q >= rays[id]["queries"].size()) continue;
                    const double x = rays[id]["queries"][q]["x"];
                    const auto& segment = profiles[id][step];
                    const bool last = step+1 == profiles[id].size();
                    if (x >= segment.begin && (x < segment.end || (last && x == segment.end))) {
                        queryIds.push_back(id); coordinates.push_back((x-segment.begin)/(segment.end-segment.begin));
                    }
                }
                const auto mixtures = batch.mixture(queryIds,coordinates);
                for (std::size_t j = 0; j < queryIds.size(); ++j) {
                    const auto& expected = rays[queryIds[j]]["queries"][q]["speed_mixture"];
                    for (std::size_t k = 0; k < mixtures[j].weights.size(); ++k) {
                        compare(mixtures[j].weights[k],expected["weights"][k],maxMixture,"mixture weights");
                        compare(mixtures[j].means[k],expected["means"][k],maxMixture,"mixture means");
                        compare(mixtures[j].scales[k],expected["scales"][k],maxMixture,"mixture scales");
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            const RenewalRayDistribution distribution(std::move(hazards[i]),rays[i]["ell"]);
            for (const auto& query : rays[i]["queries"]) {
                compare(distribution.cumulativeHazard(query["distance"]),query["cumulative_hazard"],maxCumulative,"cumulative");
                compare(distribution.transmittance(query["distance"]),query["transmittance"],maxTransmittance,"transmittance");
            }
        }
        const Json report{{"passed",true},{"backend",batch.backend()},{"checkpoint_sha256",model.checkpointSha256()},
            {"reference",argv[2]},{"rays",n},{"segments",segmentCount},{"max_rate_error",maxRate},
            {"max_mixture_parameter_error",maxMixture},{"max_cumulative_error",maxCumulative},
            {"max_transmittance_error",maxTransmittance}};
        const std::filesystem::path output(argv[4]);
        if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("cannot write batch parity report");
        stream << report.dump(2) << '\n';
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
    return 0;
}

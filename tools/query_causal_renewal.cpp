#include "macrofacet/learned/CausalRenewalModel.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: macrofacet_query_causal_renewal <model.json> <request.json> <output.json>\n";
        return 2;
    }
    try {
        using namespace mf;
        using Json = nlohmann::json;
        std::ifstream input(argv[2]); Json request; input >> request;
        std::vector<RayMeanSegment> segments;
        double begin = 0;
        for (const auto& row : request.at("features")) {
            const auto f = row.get<std::array<double,5>>();
            for (double x : f) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite profile feature");
            const double width = std::exp(f[4]), delta = f[1]-f[0];
            if (!(width > 0) || !std::isfinite(width)) throw std::invalid_argument("invalid profile width");
            segments.push_back({begin,begin+width,{-2*delta+f[2]+f[3],3*delta-2*f[2]-f[3],f[2],f[0]}});
            begin += width;
        }
        const RayMeanProfile profile(std::move(segments));
        const auto& birth = request.at("start");
        const std::string mode = birth.at("mode");
        RayStartCondition start;
        if (mode == "collision_state") {
            start.mode = RayStartMode::SurfaceOutward;
            start.outwardDerivative = birth.at("outward_derivative"); // ell/sigma * dot(omega, gradient)
        } else if (mode != "positive_exterior") throw std::invalid_argument("unknown A/B start mode");
        auto queries = request.at("queries").get<std::vector<double>>();
        for (double& q : queries) if (q > begin && q-begin < 1e-10) q = begin;
        const auto model = CausalRenewalModel::load(argv[1]);
        Json result = {{"predictions",Json::array()},{"samples",Json::array()}};
        for (const auto& p : model.query(profile,start,queries)) {
            result["predictions"].push_back({{"q",p.q},{"H",p.cumulativeHazard},
                {"log_T",-p.cumulativeHazard},{"log_hazard",p.logHazard},
                {"mixture",{{"weights",p.mixture.weights},{"means",p.mixture.means},{"scales",p.mixture.scales}}}});
        }
        if (request.contains("uniforms")) for (const auto& u : request.at("uniforms")) {
            const auto s = model.sample(profile,start,u.get<std::array<double,3>>(),request.value("tolerance",1e-9));
            result["samples"].push_back({{"hit",s.hit},{"q",s.q},{"speed",s.speed},{"intervals",s.intervals}});
        }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot write causal renewal query output");
        output << result.dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

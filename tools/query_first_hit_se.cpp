#include "macrofacet/learned/FirstHitSEModel.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    if (argc!=4 && argc!=5) {
        std::cerr << "usage: macrofacet_query_first_hit_se <model.json> <profile.json> <output.json> [trials]\n";
        return 2;
    }
    try {
        using namespace mf;
        using Json=nlohmann::json;
        std::ifstream input(argv[2]); Json request; input >> request;
        std::vector<RayMeanSegment> segments;
        double begin=0;
        for (const auto& row:request.at("features")) {
            const auto f=row.get<std::array<double,5>>();
            const double width=std::exp(f[4]);
            if (!(width>0) || !std::isfinite(width)) throw std::invalid_argument("invalid profile width");
            const double delta=f[1]-f[0];
            segments.push_back({begin,begin+width,{-2*delta+f[2]+f[3],3*delta-2*f[2]-f[3],f[2],f[0]}});
            begin+=width;
        }
        const RayMeanProfile profile(std::move(segments));
        auto queries=request.at("queries").get<std::vector<double>>();
        for (double& q:queries) if (q>begin && q-begin<1e-10) q=begin;
        const auto model=FirstHitSEModel::load(argv[1]);
        const auto predictions=model.query(profile,queries,request.value("maximum_step",0.));
        Json result={{"architecture",model.architecture()},{"predictions",Json::array()},{"samples",Json::array()}};
        for (const auto& p:predictions) {
            Json mixtures=Json::array();
            for (const auto& m:p.mixture) mixtures.push_back({{"weights",m.weights},{"means",m.means},{"scales",m.scales}});
            result["predictions"].push_back({{"q",p.q},{"log_T",p.logTransmittance},{"H",p.cumulative},
                {"log_hazard",p.logHazard},{"hit_branch_probability",p.hitBranchProbability},{"mixtures",mixtures}});
        }
        const int trials=argc==5 ? std::stoi(argv[4]) : 0;
        if (trials<0) throw std::invalid_argument("negative trial count");
        Random rng(request.value("seed",std::uint64_t(20261013)));
        for (int i=0;i<trials;++i) {
            const auto s=model.sample(profile,rng,request.value("maximum_step",0.));
            result["samples"].push_back({{"hit",s.hit},{"q",s.q},{"initial_sign",s.initialSign},{"speed",s.speed}});
        }
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot write first-hit query output");
        output << result.dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

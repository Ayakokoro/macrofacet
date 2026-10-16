#include "RenewalProfile.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
    if (argc!=6) {
        std::cerr<<"usage: macrofacet_profile_causal_first_hit config spp warm-repeats output.json baseline|wall\n";
        return 2;
    }
    try {
        at::set_num_threads(1);
        const auto setup=renewal_profile::Clock::now();
        mf::registerNanoVdbFieldTypes();
        auto config=mf::loadExperimentConfig(argv[1]);
        config.render.samplesPerPixel=std::stoi(argv[2]);
        config.render.writeFirstHitSamples=false;
        const int repeats=std::stoi(argv[3]);
        const std::string mode=argv[5];
        if (config.transportMode!="hybrid_first_hit" || config.render.samplesPerPixel<1 || repeats<1 ||
            (mode!="baseline" && mode!="wall")) throw std::invalid_argument("invalid causal profiling arguments");
        mf::prepareRenewalModel(config); mf::prepareFirstHitModel(config); mf::prepareNanoVdbField(config);
        if (!config.firstHit.model || !config.firstHit.model->causal()) throw std::invalid_argument("causal model required");
        if (config.render.environment=="unit_white") config.material.conductor.forceUnitFresnel=true;
        nlohmann::json report{{"config",argv[1]},{"backend",config.firstHit.backend},{"mode",mode},
            {"width",config.render.width},{"height",config.render.height},{"spp",config.render.samplesPerPixel},
            {"batch_size",config.firstHit.batchSize},{"cpu_threads",config.render.threadCount},
            {"setup_seconds",std::chrono::duration<double>(renewal_profile::Clock::now()-setup).count()},
            {"note","First render is cold; subsequent identical renders stay in the same process. Host times include synchronization and are not isolated GPU execution times. No extra per-stage CUDA barriers."},
            {"runs",nlohmann::json::array()}};
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output.parent_path());
        for (int run=0;run<=repeats;++run) {
            renewal_profile::reset(); renewal_profile::enabled=mode=="wall";
            const auto start=renewal_profile::Clock::now();
            auto image=mf::renderAnalyticScene(config);
            const double seconds=std::chrono::duration<double>(renewal_profile::Clock::now()-start).count();
            renewal_profile::enabled=false;
            const auto snapshot=renewal_profile::snapshot();
            nlohmann::json stages=nlohmann::json::object();
            for (const auto& [key,value]:snapshot.timings)
                stages[key]={{"seconds",value.inclusive},{"exclusive_seconds",value.exclusive},{"calls",value.calls}};
            mf::Spectrum sum=mf::Spectrum::Zero(); for (const auto& pixel:image.pixels) sum+=pixel;
            const auto& stat=image.statistics;
            report["runs"].push_back({{"phase",run==0 ? "cold" : "warm"},{"seconds",seconds},
                {"stages",stages},{"counts",snapshot.counts},{"paths",stat.paths},{"first_hits",stat.firstHits},
                {"numerical_failures",stat.numericalFailures},{"safety_cap_terminations",stat.safetyCapTerminations},
                {"image_sum",{sum[0],sum[1],sum[2]}}});
            std::ofstream(output)<<report.dump(2)<<'\n';
            std::cout<<(run==0 ? "cold" : "warm")<<" render "<<run<<": "<<seconds<<" s"<<std::endl;
        }
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

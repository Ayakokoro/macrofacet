// Fixed-variate checks against the production scalar sampler. Includes actual
// contact B states, camera A rays, and complete-flight survival probabilities.
#include "macrofacet/integrator/CausalRenewalCuda.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

using namespace mf;
using Json=nlohmann::json;
Vector3 vector(const Json& j) { return {j.at(0).get<double>(),j.at(1).get<double>(),j.at(2).get<double>()}; }
int main(int argc,char** argv) {
    try {
        if (argc<4 || argc>5) throw std::invalid_argument("usage: macrofacet_verify_causal_cuda config.json capture.json output.json [trials=64]");
        registerNanoVdbFieldTypes();
        auto c=loadExperimentConfig(argv[1]); c.renewal.backend="scalar";
        c.renewal.sampler="cumulative_inversion"; prepareRenewalModel(c);
        const RenewalMedium medium(c.field,*c.renewal.causalModel,c.renewal.profileMaximumStep,c.renewal.profileMode);
        const int trials=argc==5 ? std::stoi(argv[4]) : 64;
        if (trials<1 || trials>4096) throw std::invalid_argument("trials must be 1..4096");
        std::ifstream input(argv[2]); Json capture; input>>capture;
        std::vector<CausalCudaFlightInput> births;
        for (const auto& s:capture.at("states")) {
            if (std::abs(s.at("sigma").get<double>()-c.field.kernel.sigma())>1e-8)
                throw std::invalid_argument("capture field sigma differs from verification config");
            CausalCudaFlightInput b; b.ray={vector(s.at("origin")),vector(s.at("direction"))};
            b.birthGradient=vector(s.at("birth_gradient")); births.push_back(b);
        }
        const auto bCount=births.size();
        const auto forward=normalizedOrThrow(c.render.cameraTarget-c.render.cameraPosition);
        Vector3 right,up; orthonormalComplement(forward,right,up);
        const double scale=std::tan(c.render.verticalFovDegrees*kPi/360),aspect=double(c.render.width)/c.render.height;
        for (int y=0;y<12;++y) for (int x=0;x<12;++x) {
            CausalCudaFlightInput b;
            b.ray={c.render.cameraPosition,normalizedOrThrow(forward+right*((2*(x+.5)/12-1)*aspect*scale)+up*((1-2*(y+.5)/12)*scale))};
            const auto interval=c.field.activeDomain.intersect(b.ray);
            if (!interval.hit || !(interval.exit>interval.entry)) continue;
            b.ray.origin=(b.ray.origin+interval.entry*b.ray.direction).cwiseMax(c.field.activeDomain.minimum).cwiseMin(c.field.activeDomain.maximum);
            births.push_back(b);
        }
        std::vector<CausalCudaFlightInput> requests;
        const auto seed=[](std::size_t index) { return 20261022ULL+0x9e3779b97f4a7c15ULL*(index+1); };
        for (std::size_t i=0;i<births.size();++i) for (int j=0;j<=trials;++j) {
            auto b=births[i]; Random rng(seed(requests.size()));
            for (double& u:b.uniforms) u=rng.openUniform01();
            for (double& z:b.normals) z=rng.standardNormal();
            b.queryOnly=j==trials; requests.push_back(b);
        }
        const auto start=std::chrono::steady_clock::now();
        const auto gpu=sampleCausalRenewalCuda(c,requests);
        const double gpuSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        Json rows=Json::array(); std::mutex mutex; std::exception_ptr error;
        std::atomic<std::size_t> next{0};
        const auto run=[&] {
            try {
                for (;;) {
                    const auto state=next.fetch_add(1); if (state>=births.size()) break;
                    const auto flight=medium.beginFlight(births[state].ray,births[state].birthGradient);
                    if (!flight) throw std::runtime_error("empty verification flight");
                    int mismatches=0,hits=0,gpuHits=0;
                    double maxQ=0,maxSpeed=0,maxGradient=0,maxNormalAngle=0;
                    for (int j=0;j<trials;++j) {
                        const auto index=state*(trials+1)+j;
                        Random rng(seed(index)); const auto cpu=medium.sample(*flight,rng,c.numeric.distanceAbsoluteTolerance);
                        const auto& g=gpu[index]; hits+=cpu.hit; gpuHits+=g.hit;
                        if (cpu.hit!=g.hit) { ++mismatches; continue; }
                        if (cpu.hit) {
                            maxQ=std::max(maxQ,std::abs(cpu.distance/flight->ell-g.q));
                            maxSpeed=std::max(maxSpeed,std::abs(cpu.speed-g.speed)/(1+cpu.speed));
                            maxGradient=std::max(maxGradient,(cpu.gradient-g.gradient).norm()/(1+cpu.gradient.norm()));
                            const Vector3 gn=normalizedOrThrow(g.gradient);
                            maxNormalAngle=std::max(maxNormalAngle,
                                std::atan2(cpu.normal.cross(gn).norm(),cpu.normal.dot(gn))*180/kPi);
                        }
                    }
                    const double p=1-medium.transmittance(*flight,flight->maximumDistance());
                    const double pg=-std::expm1(-gpu[state*(trials+1)+trials].cumulativeHazard);
                    Json row{{"state",state},{"kind",state<bCount?"contact_B":"camera_A"},
                        {"cpu_probability",p},{"gpu_probability",pg},{"probability_error",std::abs(p-pg)},
                        {"cpu_hits",hits},{"gpu_hits",gpuHits},{"hit_mismatches",mismatches},
                        {"max_q_error",maxQ},{"max_speed_scaled_error",maxSpeed},{"max_gradient_scaled_error",maxGradient},
                        {"max_normal_angle_degrees",maxNormalAngle}};
                    std::lock_guard<std::mutex> lock(mutex); rows.push_back(row);
                }
            } catch (...) { std::lock_guard<std::mutex> lock(mutex); if (!error) error=std::current_exception(); }
        };
        const auto cpuStart=std::chrono::steady_clock::now();
        std::vector<std::thread> workers; for (int i=0;i<8;++i) workers.emplace_back(run);
        for (auto& worker:workers) worker.join(); if (error) std::rethrow_exception(error);
        std::sort(rows.begin(),rows.end(),[](const Json& a,const Json& b){return a.at("state").get<int>()<b.at("state").get<int>();});
        Json summary{{"sampler","cumulative_inversion"},{"B_states",bCount},{"A_states",births.size()-bCount},{"trials_per_state",trials},
            {"gpu_seconds_including_upload",gpuSeconds},
            {"cpu_seconds_8_threads",std::chrono::duration<double>(std::chrono::steady_clock::now()-cpuStart).count()},
            {"checkpoint_sha256",c.renewal.causalModel->checkpointSha256()},{"hit_mismatches",0}};
        for (const char* key:{"probability_error","max_q_error","max_speed_scaled_error","max_gradient_scaled_error","max_normal_angle_degrees"}) {
            double maximum=0; for (const auto& row:rows) maximum=std::max(maximum,row.at(key).get<double>());
            summary[key]=maximum;
        }
        int mismatch=0; for (const auto& row:rows) mismatch+=row.at("hit_mismatches").get<int>();
        summary["hit_mismatches"]=mismatch;
        const bool pass=mismatch==0 && summary["probability_error"].get<double>()<5e-6 &&
            summary["max_q_error"].get<double>()<1e-4 && summary["max_speed_scaled_error"].get<double>()<1e-4 &&
            summary["max_gradient_scaled_error"].get<double>()<1e-4 && summary["max_normal_angle_degrees"].get<double>()<.05;
        summary["passed"]=pass;
        const std::filesystem::path output=argv[3]; if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        std::ofstream file(output); file<<std::setprecision(17)<<Json{{"summary",summary},{"states",rows}}.dump(2)<<'\n';
        if (!file) throw std::runtime_error("failed writing CUDA verification report");
        std::cout<<summary.dump(2)<<'\n'; return pass ? 0 : 2;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

// Distribution checks for the variable-length null-collision sampler. RNG
// streams differ from inversion, so fixed-variate hit equality is not a test.
#include "macrofacet/integrator/CausalRenewalCuda.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

using namespace mf;
using Json=nlohmann::json;
namespace {
Vector3 vector(const Json& j) { return {j.at(0).get<double>(),j.at(1).get<double>(),j.at(2).get<double>()}; }
std::uint64_t seed(std::size_t i) { return 20261022ULL+0x9e3779b97f4a7c15ULL*(i+1); }
double ks(std::vector<double> a,std::vector<double> b) {
    std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
    std::size_t i=0,j=0; double result=0;
    while (i<a.size() || j<b.size()) {
        const double x=std::min(i<a.size()?a[i]:INFINITY,j<b.size()?b[j]:INFINITY);
        while (i<a.size() && a[i]<=x) ++i;
        while (j<b.size() && b[j]<=x) ++j;
        result=std::max(result,std::abs(double(i)/a.size()-double(j)/b.size()));
    }
    return result;
}
// Union of two one-sample DKW bounds: valid for unequal sample counts and
// conservative across all per-state comparisons (family error <= 0.001).
double ksLimit(std::size_t a,std::size_t b,std::size_t comparisons) {
    const double l=std::log(4.*comparisons/.001);
    return std::sqrt(l/(2*a))+std::sqrt(l/(2*b));
}
}
int main(int argc,char** argv) {
    try {
        if (argc<4 || argc>6) throw std::invalid_argument(
            "usage: macrofacet_verify_causal_delta config.json capture.json output.json [trials=2048] [state_stride=1]");
        registerNanoVdbFieldTypes(); auto c=loadExperimentConfig(argv[1]);
        c.renewal.backend="cuda_wavefront"; c.renewal.sampler="segmented_delta_tracking"; prepareRenewalModel(c);
        const int trials=argc>4 ? std::stoi(argv[4]) : 2048,stride=argc>5 ? std::stoi(argv[5]) : 1;
        if (trials<64 || trials>16384 || stride<1) throw std::invalid_argument("trials must be 64..16384; stride positive");
        std::ifstream file(argv[2]); Json capture; file>>capture;
        std::vector<CausalCudaFlightInput> births;
        for (std::size_t i=0;i<capture.at("states").size();i+=stride) {
            const auto& s=capture.at("states").at(i);
            if (std::abs(s.at("sigma").get<double>()-c.field.kernel.sigma())>1e-8)
                throw std::invalid_argument("capture sigma differs from verification config");
            CausalCudaFlightInput b; b.ray={vector(s.at("origin")),vector(s.at("direction"))};
            b.birthGradient=vector(s.at("birth_gradient")); births.push_back(b);
        }
        const auto bCount=births.size();
        const Vector3 forward=normalizedOrThrow(c.render.cameraTarget-c.render.cameraPosition);
        Vector3 right,up; orthonormalComplement(forward,right,up);
        const double scale=std::tan(c.render.verticalFovDegrees*kPi/360),aspect=double(c.render.width)/c.render.height;
        for (int i=0;i<144;i+=stride) {
            const int x=i%12,y=i/12; CausalCudaFlightInput b;
            b.ray={c.render.cameraPosition,normalizedOrThrow(forward+right*((2*(x+.5)/12-1)*aspect*scale)+up*((1-2*(y+.5)/12)*scale))};
            const auto interval=c.field.activeDomain.intersect(b.ray);
            if (!interval.hit || !(interval.exit>interval.entry)) continue;
            b.ray.origin=(b.ray.origin+interval.entry*b.ray.direction).cwiseMax(c.field.activeDomain.minimum).cwiseMin(c.field.activeDomain.maximum);
            births.push_back(b);
        }
        const auto requests=[&](bool delta) {
            std::vector<CausalCudaFlightInput> result; result.reserve(births.size()*(trials+1));
            for (const auto& birth:births) for (int j=0;j<=trials;++j) {
                auto b=birth; const auto id=seed(result.size());
                Random rng(delta ? id : id^0x728d83bc4fa93851ULL);
                for (double& u:b.uniforms) u=rng.openUniform01();
                for (double& z:b.normals) z=rng.standardNormal();
                b.trackingSeed=id^0xda942042e4dd58b5ULL; b.queryOnly=j==trials; result.push_back(b);
            }
            return result;
        };
        const auto start=std::chrono::steady_clock::now();
        auto delta=sampleCausalRenewalCuda(c,requests(true));
        const double deltaSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"delta samples finished: "<<deltaSeconds<<" seconds\n"<<std::flush;
        auto inverseConfig=c; inverseConfig.renewal.sampler="cumulative_inversion";
        const auto inverseStart=std::chrono::steady_clock::now();
        const auto inverse=sampleCausalRenewalCuda(inverseConfig,requests(false));
        const double inverseSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-inverseStart).count();
        const RenewalMedium medium(c.field,*c.renewal.causalModel,c.renewal.profileMaximumStep,c.renewal.profileMode);
        Json rows=Json::array(); bool passed=true; double maxZ=0,maxProbabilityError=0,maxKsFraction=0;
        std::uint64_t candidates=0,nulls=0,bounds=0,deltaHits=0,inverseHits=0;
        int skippedNormals=0; double expectedHits=0,hitVariance=0;
        for (std::size_t i=0;i<births.size();++i) {
            const auto flight=medium.beginFlight(births[i].ray,births[i].birthGradient);
            if (!flight) throw std::runtime_error("empty reference flight");
            std::vector<double> q;
            for (int k=1;k<=32;++k) q.push_back(flight->mean.maximumX()*k/32.);
            for (double x:{1e-5,1e-4,.001,.01,.05,.125,.25,.5}) if (x<flight->mean.maximumX()) q.push_back(x);
            std::sort(q.begin(),q.end());
            const auto prediction=c.renewal.causalModel->query(flight->mean,flight->start,q);
            std::array<std::vector<double>,5> a,b; // distance (including escape), speed, normal XYZ
            for (int j=0;j<trials;++j) {
                const auto index=i*(trials+1)+j;
                const auto append=[&](const CausalCudaFlightOutput& sample,auto& values) {
                    values[0].push_back(sample.hit ? sample.q : INFINITY);
                    if (sample.hit) {
                        if (!sample.gradient.allFinite() || !(sample.gradient.dot(births[i].ray.direction)<0) ||
                            !(sample.speed>0) || !(sample.q>=0 && sample.q<=flight->mean.maximumX()))
                            throw std::runtime_error("invalid accepted collision");
                        values[1].push_back(sample.speed); const Vector3 normal=normalizedOrThrow(sample.gradient);
                        for (int axis=0;axis<3;++axis) values[axis+2].push_back(normal[axis]);
                    }
                };
                append(delta[index],a); append(inverse[index],b);
                candidates+=delta[index].candidates; nulls+=delta[index].nullCollisions; bounds+=delta[index].boundIntervals;
                if (delta[index].candidates!=delta[index].nullCollisions+std::uint64_t(delta[index].hit))
                    throw std::runtime_error("candidate accounting mismatch");
            }
            deltaHits+=a[1].size(); inverseHits+=b[1].size();
            std::sort(a[0].begin(),a[0].end());
            double zMax=0; Json cdf=Json::array();
            for (std::size_t k=0;k<q.size();++k) {
                const double p=-std::expm1(-prediction[k].cumulativeHazard);
                const auto hits=std::upper_bound(a[0].begin(),a[0].end(),q[k])-a[0].begin();
                const double z=std::abs(hits-trials*p)/std::sqrt(trials*p*(1-p)+1);
                zMax=std::max(zMax,z); cdf.push_back({{"q",q[k]},{"reference_cdf",p},{"observed_cdf",double(hits)/trials}});
            }
            const double p=-std::expm1(-prediction.back().cumulativeHazard);
            const double pg=-std::expm1(-delta[i*(trials+1)+trials].cumulativeHazard);
            const double probabilityError=std::abs(p-pg);
            maxProbabilityError=std::max(maxProbabilityError,probabilityError); maxZ=std::max(maxZ,zMax);
            expectedHits+=trials*p; hitVariance+=trials*p*(1-p);
            Json metrics=Json::object(); bool statePass=zMax<6.5 && probabilityError<5e-6;
            const std::array<const char*,5> names{"distance","speed","normal_x","normal_y","normal_z"};
            if (a[1].size()<64 || b[1].size()<64) ++skippedNormals;
            for (int k=0;k<5;++k) {
                if (k && (a[k].size()<64 || b[k].size()<64)) continue;
                const double error=ks(a[k],b[k]),limit=ksLimit(a[k].size(),b[k].size(),births.size()*5);
                metrics[names[k]]={{"ks",error},{"limit",limit}};
                statePass&=error<=limit; maxKsFraction=std::max(maxKsFraction,error/limit);
            }
            passed&=statePass;
            rows.push_back({{"state",i},{"kind",i<bCount?"contact_B":"camera_A"},{"reference_hit_probability",p},
                {"delta_hits",a[1].size()},{"inversion_hits",b[1].size()},{"max_cdf_z",zMax},
                {"probability_error",probabilityError},{"distribution_checks",metrics},{"cdf",cdf},{"passed",statePass}});
        }
        const double aggregateZ=std::abs(double(deltaHits)-expectedHits)/std::sqrt(hitVariance+1);
        passed&=aggregateZ<6.5;
        Json summary{{"passed",passed},{"sampler","segmented_delta_tracking"},{"B_states",bCount},
            {"A_states",births.size()-bCount},{"trials_per_state",trials},{"state_stride",stride},
            {"delta_seconds_including_upload",deltaSeconds},{"inversion_seconds_including_upload",inverseSeconds},
            {"max_reference_cdf_z",maxZ},{"aggregate_hit_z",aggregateZ},{"max_full_hazard_probability_error",maxProbabilityError},
            {"max_ks_fraction_of_limit",maxKsFraction},{"states_with_insufficient_hits_for_normal_ks",skippedNormals},
            {"delta_hits",deltaHits},{"inversion_hits",inverseHits},{"candidates",candidates},{"null_collisions",nulls},
            {"bound_intervals",bounds},{"checkpoint_sha256",c.renewal.causalModel->checkpointSha256()},
            {"note","Statistical tests against independent inversion samples and scalar CDF; not pixelwise or fixed-variate equality. Upper-bound violations throw during sampling."}};
        const std::filesystem::path output=argv[3];
        if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        std::ofstream out(output); out<<Json{{"summary",summary},{"states",rows}}.dump(2)<<'\n';
        if (!out) throw std::runtime_error("cannot write verification report");
        std::cout<<summary.dump(2)<<'\n'; return passed ? 0 : 2;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

// Replay production neural camera paths, then compare fixed surface-born flights.
// This diagnostic does not change the renderer or its random-number streams.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace {
using namespace mf;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Sample = std::array<double,5>; // status, physical distance, gx, gy, gz
constexpr std::uint64_t stride = 0x9e3779b97f4a7c15ULL;
Json vec(const Vector3& v) { return {v.x(),v.y(),v.z()}; }
Vector3 vec(const Json& v) { return {v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>()}; }
Json read(const std::filesystem::path& p) {
    std::ifstream f(p); if (!f) throw std::runtime_error("cannot read "+p.string());
    Json j; f>>j; return j;
}
void write(const std::filesystem::path& p,const Json& j) {
    std::ofstream f(p); if (!f) throw std::runtime_error("cannot write "+p.string());
    f<<std::setprecision(17)<<j.dump(2)<<'\n';
    if (!f) throw std::runtime_error("write failed "+p.string());
}
Spectrum environment(const Vector3& d,const ExperimentConfig& c) {
    if (c.render.environment=="unit_white") return Spectrum::Ones();
    if (c.render.environment!="directional_gradient") throw std::invalid_argument("unsupported environment");
    const double up=.5*(normalizedOrThrow(d).z()+1);
    return { .15+.85*up,.2+.7*up,.35+.55*up };
}
struct Birth {
    Ray ray;
    Vector3 gradient;
    Spectrum throughput;
    int pixelX=0,pixelY=0,sample=0,depth=0;
    bool nextHit=false;
    double nextDistance=0;
};

// Deliberately follows traceRenewalFlights and renderRow draw-for-draw. The
// recorded pixel averages are checked against the saved production PFM later.
Spectrum trace(Ray ray,const ExperimentConfig& c,const RenewalMedium& medium,
    Random& rng,int px,int py,int sample,std::vector<Birth>& births) {
    ray.direction=normalizedOrThrow(ray.direction);
    const auto domain=c.field.activeDomain.intersect(ray);
    if (!domain.hit || !(domain.exit>domain.entry)) return environment(ray.direction,c);
    ray.origin+=domain.entry*ray.direction;
    ray.origin=ray.origin.cwiseMax(c.field.activeDomain.minimum).cwiseMin(c.field.activeDomain.maximum);
    Spectrum throughput=Spectrum::Ones();
    std::optional<Vector3> gradient;
    for (int depth=0;depth<c.render.safetyDepthCap;++depth) {
        const auto flight=medium.beginFlight(ray,gradient);
        const auto hit=flight ? medium.sample(*flight,rng,c.numeric.distanceAbsoluteTolerance) : RenewalSurfaceSample{};
        if (depth>0 && flight) births.push_back({ray,*gradient,throughput,px,py,sample,depth,hit.hit,hit.distance});
        if (!hit.hit) return throughput.cwiseProduct(environment(ray.direction,c));
        throughput=throughput.cwiseProduct(conductorFresnel(-ray.direction.dot(hit.normal),c.material.conductor));
        ray={hit.position,normalizedOrThrow(reflectTravelDirection(ray.direction,hit.normal))};
        gradient=hit.gradient;
        if (!(ray.direction.dot(*gradient)>0)) throw std::runtime_error("non-outward replay flight");
        if (depth+1>=c.render.rouletteStartDepth) {
            const double p=std::clamp(throughput.maxCoeff(),.05,.95);
            if (rng.openUniform01()>=p) return Spectrum::Zero();
            throughput/=p;
        }
    }
    return Spectrum::Zero();
}

void capture(const Json& request,const ExperimentConfig& c,const RenewalMedium& medium,
    const std::filesystem::path& output) {
    const int grid=request.at("pixels_per_axis"), limit=request.at("states_per_region_depth");
    if (grid<1 || grid>64 || limit<1) throw std::invalid_argument("invalid capture sizes");
    const auto forward=normalizedOrThrow(c.render.cameraTarget-c.render.cameraPosition);
    Vector3 right,up; orthonormalComplement(forward,right,up);
    const double scale=std::tan(c.render.verticalFovDegrees*kPi/360),aspect=double(c.render.width)/c.render.height;
    Json selected=Json::array(),strata=Json::array(),pixels=Json::array();
    for (const auto& region:request.at("regions")) {
        const auto box=region.at("xyxy").get<std::array<int,4>>();
        if (box[0]<0 || box[1]<0 || box[2]>c.render.width || box[3]>c.render.height ||
            box[2]-box[0]<grid || box[3]-box[1]<grid) throw std::invalid_argument("invalid ROI");
        std::vector<Birth> pool;
        for (int gy=0;gy<grid;++gy) for (int gx=0;gx<grid;++gx) {
            const int px=box[0]+int((gx+.5)*(box[2]-box[0])/grid);
            const int py=box[1]+int((gy+.5)*(box[3]-box[1])/grid);
            const auto pixel=std::uint64_t(py)*c.render.width+px;
            Random rng(c.seed+stride*(pixel+1));
            Spectrum sum=Spectrum::Zero();
            for (int s=0;s<c.render.samplesPerPixel;++s) {
                const double u=(2*((px+rng.openUniform01())/c.render.width)-1)*aspect*scale;
                const double v=(1-2*((py+rng.openUniform01())/c.render.height))*scale;
                const Ray ray{c.render.cameraPosition,normalizedOrThrow(forward+u*right+v*up)};
                sum+=trace(ray,c,medium,rng,px,py,s,pool);
            }
            pixels.push_back({{"region",region.at("name")},{"pixel",{px,py}},
                {"radiance",vec(Spectrum(sum/c.render.samplesPerPixel))}});
        }
        // Select without consulting next-hit outcomes or comparison errors.
        // First B and later B flights are separate strata, with population weights.
        for (int group=0;group<2;++group) {
            std::vector<std::size_t> indices;
            for (std::size_t i=0;i<pool.size();++i) if ((pool[i].depth==1)==(group==0)) indices.push_back(i);
            const auto population=indices.size();
            std::mt19937_64 selection(request.at("seed").get<std::uint64_t>()+stride*(strata.size()+1));
            std::shuffle(indices.begin(),indices.end(),selection);
            indices.resize(std::min(indices.size(),static_cast<std::size_t>(limit)));
            strata.push_back({{"region",region.at("name")},{"kind",region.at("kind")},
                {"depth_group",group==0 ? "first_B" : "later_B"},
                {"population",population},{"selected",indices.size()},
                {"camera_paths",grid*grid*c.render.samplesPerPixel}});
            for (auto i:indices) {
                const auto& b=pool[i]; const auto flight=medium.beginFlight(b.ray,b.gradient);
                const auto& first=flight->mean.segments().front();
                Vector3 e1,e2; orthonormalComplement(b.ray.direction,e1,e2);
                selected.push_back({{"id",selected.size()},{"region",region.at("name")},{"kind",region.at("kind")},
                    {"depth_group",group==0 ? "first_B" : "later_B"},{"population_weight",double(population)/indices.size()},
                    {"pixel",{b.pixelX,b.pixelY}},{"camera_sample",b.sample},{"completed_collisions",b.depth},
                    {"origin",vec(b.ray.origin)},{"direction",vec(b.ray.direction)},{"birth_gradient",vec(b.gradient)},
                    {"throughput",vec(b.throughput)},{"e1",vec(e1)},{"e2",vec(e2)},
                    {"sigma",flight->sigma},{"ell",flight->ell},{"horizon",flight->mean.maximumX()},
                    {"b0",first.value(0)},{"bprime0",first.derivative(0)},
                    {"beta_g",flight->start.outwardDerivative},
                    {"derivative_residual",flight->start.outwardDerivative-first.derivative(0)},
                    {"birth_mean_gradient",vec(flight->birthMeanGradient)},
                    {"original_next_hit",b.nextHit},{"original_next_distance",b.nextDistance}});
            }
        }
        std::cout<<region.at("name")<<": captured "<<pool.size()<<" B flights"<<std::endl;
    }
    write(output/"capture.json",{{"strata",strata},{"states",selected},{"pixels",pixels},
        {"selection","uniform without replacement within region and first/later B stratum; independent of next outcomes"},
        {"source","exact replay of neural per_pixel_v1, all original samples of selected pixels"}});
}

void binary(const std::filesystem::path& path,const std::vector<Sample>& rows) {
    static_assert(sizeof(Sample)==5*sizeof(double));
    const std::uint16_t endian=1;
    if (*reinterpret_cast<const unsigned char*>(&endian)!=1) throw std::runtime_error("little-endian output required");
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(rows.data()),static_cast<std::streamsize>(rows.size()*sizeof(Sample)));
    if (!f) throw std::runtime_error("binary write failed");
}
Sample record(int status,double distance,const Vector3& g=Vector3::Zero()) {
    if (status==1 && (!g.allFinite() || !(g.squaredNorm()>0))) throw std::runtime_error("invalid collision gradient");
    return {double(status),distance,g.x(),g.y(),g.z()};
}
Vector3 meanAt(const GPSSField& field,const RenewalFlight& flight,double q) {
    const auto& segments=flight.mean.segments();
    auto it=std::lower_bound(segments.begin(),segments.end(),q,[](const auto& s,double x){return s.end<x;});
    if (it==segments.end()) it=std::prev(segments.end());
    const Point3 point=(flight.ray.origin+q*flight.ell*flight.ray.direction)
        .cwiseMax(field.activeDomain.minimum).cwiseMin(field.activeDomain.maximum);
    return field.mean->evaluateInCell(point,flight.ray.origin+(.5*(it->begin+it->end)*flight.ell)*flight.ray.direction).gradient;
}

Json compare(const Json& state,const Json& request,const ExperimentConfig& c,
    const RenewalMedium& neural,const std::filesystem::path& output) {
    const auto began=Clock::now();
    const int id=state.at("id"),trials=request.at("trials_per_state");
    const Ray ray{vec(state.at("origin")),vec(state.at("direction"))};
    const auto birth=vec(state.at("birth_gradient"));
    const auto flight=neural.beginFlight(ray,birth);
    const auto& model=*c.renewal.causalModel;
    const FunctionSpaceGpisMedium gpis(c.field,{request.at("gpis_sample_points"),request.at("gpis_step_size")});
    const auto seed=request.at("seed").get<std::uint64_t>()+stride*(id+1);
    std::vector<Sample> nn,gp; nn.reserve(trials); gp.reserve(trials);
    Random nr(seed); TungstenRandom gr(seed^0x67706973ULL);
    std::array<int,4> nc{},gc{};
    std::vector<double> targets;
    std::vector<std::size_t> accepted;
    for (int t=0;t<trials;++t) {
        const auto n=neural.sample(*flight,nr,c.numeric.distanceAbsoluteTolerance);
        const auto g=gpis.sample(ray,TungstenSurfaceState{ray.origin,0,birth},gr);
        const int ns=n.hit ? 1 : 0,gs=static_cast<int>(g.status);
        ++nc[ns];++gc[gs];
        nn.push_back(record(ns,n.distance,n.gradient));
        gp.push_back(record(gs,g.distance,g.surface.gradient));
        if (gs==1) {
            // Float Tungsten geometry can move its endpoint by a few ulps.
            const double q=g.distance/flight->ell;
            if (q < -1e-5 || q>flight->mean.maximumX()+1e-5) throw std::runtime_error("GPIS distance outside shared flight");
            targets.push_back(std::clamp(q,0.,flight->mean.maximumX()));
            accepted.push_back(t);
        }
    }
    std::vector<double> curveQ;
    for (int i=0;i<=256;++i) curveQ.push_back(flight->mean.maximumX()*i/256.);
    auto curve=model.query(flight->mean,flight->start,curveQ);
    Json curveJson=Json::array();
    for (const auto& p:curve) curveJson.push_back({p.q,std::exp(-p.cumulativeHazard)});

    // Hold the GPIS distance distribution fixed. Paired variants use the same
    // mixture variates and transverse Gaussian noise, changing one condition.
    const auto predictions=model.query(flight->mean,flight->start,targets);
    std::array<std::vector<Sample>,4> matched;
    for (std::size_t j=0;j<accepted.size();++j) {
        const double q=targets[j],distance=q*flight->ell;
        const auto& g=gp[accepted[j]];
        const Vector3 gpGradient(g[2],g[3],g[4]);
        const double gpDerivative=ray.direction.dot(gpGradient);
        if (!(gpDerivative<0)) throw std::runtime_error("GPIS accepted a non-downcrossing");
        Random speedRng(seed+stride*(accepted[j]+123));
        const double w=predictions[j].mixture.sample(speedRng);
        const Vector3 mean=meanAt(c.field,*flight,q);
        const double nd=-flight->sigma/flight->ell*w;
        for (int variant=0;variant<4;++variant) {
            Random noise=speedRng;
            const double derivative=variant<2 ? nd : gpDerivative;
            const std::optional<Vector3> previous=variant%2==0 ? std::optional<Vector3>(birth) : std::nullopt;
            const auto gradient=sampleRenewalGradient(c.field.kernel,ray.direction,q,derivative,
                mean,flight->birthMeanGradient,previous,noise);
            matched[variant].push_back(record(1,distance,gradient));
        }
    }
    const std::string prefix="state_"+std::to_string(id)+"_";
    binary(output/(prefix+"neural.bin"),nn); binary(output/(prefix+"gpis.bin"),gp);
    const std::array<std::string,4> names{"joint_at_gpis_distance","joint_at_gpis_distance_no_memory",
        "gpis_speed_full_memory","gpis_speed_no_memory"};
    for (int i=0;i<4;++i) binary(output/(prefix+names[i]+".bin"),matched[i]);
    Json result{{"id",id},{"region",state.at("region")},{"trials",trials},{"neural_counts",nc},{"gpis_counts",gc},
        {"predicted_escape",curveJson.back().at(1)},{"survival_curve_q_T",curveJson},
        {"matched_samples",accepted.size()},{"seconds",std::chrono::duration<double>(Clock::now()-began).count()}};
    write(output/(prefix+"result.json"),result);
    // Ready-to-run independent training-reference experiment on the SAME B state.
    const int coarseCells=std::max(1,int(std::ceil(flight->mean.maximumX()/.015625)));
    const double coarse=flight->mean.maximumX()/coarseCells;
    const auto source=read(request.at("source_config").get<std::string>());
    Json reference{{"schema_version",1},{"seed",seed},
        {"output_directory",(output/(prefix+"reference")).generic_string()},
        {"first_passage",{
            {"process",{{"mean_field",source.at("field")}}},
            {"initial_condition",{{"type","collision_state"},{"rays",Json::array({{
                {"id",prefix+"B"},{"origin",vec(ray.origin)},{"direction",vec(ray.direction)},{"gradient",vec(birth)}}})}}},
            {"grid",{{"max_time",flight->mean.maximumX()},{"step_sizes",{coarse,coarse/2}}}},
            {"profile",{{"maximum_step",c.renewal.profileMaximumStep}}},
            {"curve",{{"bins",256}}},
            {"monte_carlo",{{"trajectories",4096},{"thread_count",4},{"write_raw_samples",true}}},
            {"sampler",{{"type","collision_state_auto"},{"minimum_step",.0009765625},
                {"crossing_tolerance",1e-9},{"bridge_sigma_margin",6},{"max_refinement_depth",16}}},
            {"kernels",Json::array({{{"id","se"},{"type","squared_exponential"},{"parameterization","unit_length"},
                {"variance",flight->sigma*flight->sigma},{"length_scale",flight->ell}}})}}}};
    write(output/(prefix+"reference_input.json"),reference);
    return result;
}
}

int main(int argc,char** argv) {
    if (argc!=2 && argc!=3) { std::cerr<<"usage: macrofacet_compare_contact_b request.json [capture|compare]\n";return 2; }
    try {
        const auto request=read(argv[1]);
        const std::string phase=argc==3 ? argv[2] : "all";
        if (phase!="all" && phase!="capture" && phase!="compare") throw std::invalid_argument("unknown phase");
        const std::filesystem::path out=request.at("output_directory").get<std::string>();
        const int workers=request.at("thread_count"),trials=request.at("trials_per_state");
        if (workers<1 || workers>64 || trials<1 || trials>1000000) throw std::invalid_argument("invalid comparison sizes");
        registerNanoVdbFieldTypes();
        auto c=loadExperimentConfig(request.at("source_config").get<std::string>());
        if (c.transportMode!="neural_causal_renewal") throw std::invalid_argument("requires neural_causal_renewal");
        prepareRenewalModel(c);prepareNanoVdbField(c);
        const RenewalMedium neural(c.field,*c.renewal.causalModel,c.renewal.profileMaximumStep,c.renewal.profileMode);
        const Json identity{{"request",request},{"source_config",read(request.at("source_config").get<std::string>())},
            {"checkpoint_sha256",c.renewal.causalModel->checkpointSha256()},
            {"binary_layout","little-endian float64 rows: status,physical_distance,gx,gy,gz"},
            {"status_codes",{{"escape",0},{"accepted_hit",1},{"rejected_upcrossing",2},{"invalid_gradient",3}}}};
        std::filesystem::create_directories(out);
        if (std::filesystem::exists(out/"identity.json")) {
            if (read(out/"identity.json")!=identity) throw std::invalid_argument("output identity changed; choose a new directory");
        } else write(out/"identity.json",identity);
        writeResolvedConfig(c,out/"resolved_source.json");
        if (phase!="compare" && !std::filesystem::exists(out/"capture.json")) capture(request,c,neural,out);
        if (phase=="capture") return 0;
        const auto captured=read(out/"capture.json");const auto& states=captured.at("states");
        std::vector<Json> results(states.size());
        std::atomic<std::size_t> next{0},completed{0};std::atomic<bool> failed{false};
        std::mutex lock;std::exception_ptr error;const auto began=Clock::now();
        const auto work=[&] {
            try {
                while (!failed) {
                    const auto i=next.fetch_add(1);if (i>=states.size()) break;
                    const auto cached=out/("state_"+std::to_string(i)+"_result.json");
                    results[i]=std::filesystem::exists(cached) ? read(cached) : compare(states[i],request,c,neural,out);
                    const auto n=completed.fetch_add(1)+1;
                    if (n%16==0 || n==states.size()) {
                        std::lock_guard<std::mutex> guard(lock);
                        std::cout<<"compared "<<n<<'/'<<states.size()<<" B states"<<std::endl;
                    }
                }
            } catch (...) { failed=true;std::lock_guard<std::mutex> guard(lock);if (!error) error=std::current_exception(); }
        };
        std::vector<std::thread> pool;
        for (int i=0;i<workers;++i) pool.emplace_back(work);
        for (auto& t:pool) t.join();if (error) std::rethrow_exception(error);
        write(out/"comparison.json",{{"states",results},{"seconds",std::chrono::duration<double>(Clock::now()-began).count()},
            {"scope","fixed states drawn from actual neural paths; not a replacement-render attribution"}});
        std::cout<<"completed "<<out.string()<<std::endl;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}

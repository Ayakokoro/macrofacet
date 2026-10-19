// Compare production first-flight laws, without reflection or continuation.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/FirstHitSEMedium.h"
#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
using namespace mf;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
struct Method { std::string id, family; double step = 0; };
struct CameraRay {
    Ray ray;
    Point3 entry;
    std::uint64_t pixel;
    double entryDistance, length, ell;
};
Json vec(const Vector3& v) { return Json::array({v.x(),v.y(),v.z()}); }
void writeJson(const std::filesystem::path& path, const Json& value) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << std::setw(2) << value << '\n';
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: macrofacet_compare_camera_first_hit <comparison.json>\n";
        return 2;
    }
    try {
        const auto start = Clock::now();
        std::ifstream requestFile(argv[1]);
        const Json request = Json::parse(requestFile);
        const std::filesystem::path output = request.at("output_directory").get<std::string>();
        const int trials = request.at("trials_per_ray").get<int>();
        const int threads = request.value("thread_count",8);
        const auto seed = request.at("seed").get<std::uint64_t>();
        const auto& pixels = request.at("pixels");
        if (trials < 1 || trials > 1000000 || threads < 1 || threads > 64 || pixels.empty())
            throw std::invalid_argument("invalid trials, thread count, or pixel selection");
        // Never silently overwrite samples from an earlier run.
        if (std::filesystem::exists(output/"request.json"))
            throw std::invalid_argument("output directory already contains a comparison; choose a new directory");
        std::filesystem::create_directories(output);
        writeJson(output/"request.json",request);
        registerNanoVdbFieldTypes();
        auto config = loadExperimentConfig(request.at("source_config").get<std::string>());
        prepareNanoVdbField(config);
        if (!config.field.kernel.supportsAnalyticConditionalTransport() ||
            config.material.ndfFamily != NdfFamily::GeneralizedGaussian)
            throw std::invalid_argument("comparison requires SE and generalized Gaussian material");
        writeResolvedConfig(config,output/"resolved_source.json");
        auto localMaterial = config.material;
        localMaterial.gpModel = GpModel::LocalTangent;
        const NarrowBandMedium local(config.field,localMaterial,config.mediumDensity,
            config.densityMajorantGrid,config.material.gpModel == GpModel::LocalTangent ?
                config.preparedAreaMajorant : std::nullopt);
        const ConditionalMedium conditional(config.field);
        const int samplePoints = request.value("gpis_sample_points",8);
        const std::vector<Method> methods{
            {"local","classic_local"},{"global_conditional","global_conditional"},
            {"gpis_1","gpis_function_space_tungsten",1.0},
            {"gpis_0.1","gpis_function_space_tungsten",0.1},
            {"gpis_0.01","gpis_function_space_tungsten",0.01},
            {"renewal_1","global_conditional_renewal_plus",1.0},
            {"renewal_0.1","global_conditional_renewal_plus",0.1},
            {"renewal_0.01","global_conditional_renewal_plus",0.01}};
        const auto forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
        Vector3 right,up;
        orthonormalComplement(forward,right,up);
        const double scale = std::tan(config.render.verticalFovDegrees*kPi/360.0);
        const double aspect = double(config.render.width)/config.render.height;
        std::vector<CameraRay> rays;
        Json rayMetadata = Json::array();
        for (const auto& pixel : pixels) {
            const int px = pixel.at(0), py = pixel.at(1);
            if (px < 0 || py < 0 || px >= config.render.width || py >= config.render.height)
                throw std::invalid_argument("pixel outside camera image");
            const double u = (2*((px+0.5)/config.render.width)-1)*aspect*scale;
            const double v = (1-2*((py+0.5)/config.render.height))*scale;
            const Ray ray{config.render.cameraPosition,normalizedOrThrow(forward+u*right+v*up)};
            const auto interval = config.field.activeDomain.intersect(ray);
            if (!interval.hit || !(interval.exit > interval.entry))
                throw std::invalid_argument("selected camera ray misses activeDomain");
            const Point3 entry = (ray.origin+interval.entry*ray.direction)
                .cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
            const auto flight = local.beginFlight(local.startExternal(entry,ray.direction));
            const double length = flight.maximumAgeInDomain();
            const double ell = 1/std::sqrt(ray.direction.dot(config.field.kernel.metric()*ray.direction));
            const auto id = rays.size();
            rays.push_back({ray,entry,std::uint64_t(py)*config.render.width+px,interval.entry,length,ell});
            rayMetadata.push_back({{"id",id},{"pixel",pixel},{"origin",vec(ray.origin)},
                {"direction",vec(ray.direction)},{"entry",vec(entry)},
                {"entry_distance",interval.entry},{"length",length},{"ell",ell},
                {"entry_mean_sigma",config.field.mean->evaluate(entry).value/config.field.kernel.sigma()}});
            std::ofstream profile(output/("ray_"+std::to_string(id)+"_profile.csv"));
            profile << std::setprecision(17) << "distance,mean,mean_derivative,mean_gradient_norm,local_hazard\n";
            for (int j=0; j<=2048; ++j) {
                const double s = length*j/2048.0;
                const auto jet = config.field.mean->evaluate(entry+s*ray.direction);
                const auto hazard = flight.evaluate(s).hazard;
                const double value = hazard.status == NumericStatus::ExactZero ? 0 : std::exp(hazard.logValue);
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite profile hazard");
                profile << s << ',' << jet.value << ',' << jet.gradient.dot(ray.direction)
                    << ',' << jet.gradient.norm() << ',' << value << '\n';
            }
        }
        Json summary{{"source_config",request.at("source_config")},{"seed",seed},
            {"trials_per_ray",trials},{"thread_count",threads},{"sigma",config.field.kernel.sigma()},
            {"rays",rayMetadata},{"methods",Json::array()},{"samples",Json::array()},
            {"pixel_convention","zero-based fixed pixel centers; same camera frame as renderer"},
            {"distance_origin","activeDomain entry, camera vacuum excluded"},
            {"status_codes",{{"escape",0},{"accepted_hit",1},{"rejected_upcrossing",2},{"invalid_gradient",3}}},
            {"rng","first_hit_split_v1, event stream 1 shared seed across methods; local gradient stream 2; GC/renewal use event stream for gradient; GPIS PCG seeded by event.integer()"},
            {"gpis_notes","production Tungsten renewal+ / gp_discrete / conditioned_gaussian / skip_space=0; unconditional initial batch, longitudinal secant gradient; float camera geometry"},
            {"survival_notes","A detected crossing, including a rejected upcrossing, terminates survival; invalid samples are reported, never treated as escapes. Gradient CDFs use accepted downcrossings."}};
        for (const auto& m : methods)
            summary["methods"].push_back({{"id",m.id},{"family",m.family},{"world_step",m.step},
                {"gpis_sample_points",m.family == "gpis_function_space_tungsten" ? samplePoints : 0}});
        writeJson(output/"metadata.json",summary);
        const double preparationSeconds = std::chrono::duration<double>(Clock::now()-start).count();
        std::cout << "prepared " << rays.size() << " rays, " << methods.size() << " methods; "
                  << trials << " trials each, " << threads << " workers" << std::endl;
        const std::size_t count = rays.size()*methods.size();
        std::vector<Json> results(count);
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr failure;
        std::mutex lock;
        const auto sampleStart = Clock::now();
        const auto worker = [&] {
            try {
                for (;;) {
                    if (failed.load()) break;
                    const auto job = next.fetch_add(1);
                    if (job >= count) break;
                    const auto id = job/methods.size();
                    const auto& r = rays[id];
                    const auto& m = methods[job%methods.size()];
                    const auto begin = Clock::now();
                    const auto localFlight = local.beginFlight(local.startExternal(r.entry,r.ray.direction));
                    const GlobalConditionalRenewalPlusMedium renewal(config.field,m.step>0 ? m.step : 1);
                    const FunctionSpaceGpisMedium gpis(config.field,{samplePoints,m.step>0 ? m.step : 0.01});
                    const std::string filename = "ray_"+std::to_string(id)+"_"+m.id+".csv";
                    std::ofstream samples(output/filename);
                    if (!samples) throw std::runtime_error("cannot write samples");
                    samples << std::setprecision(17) << "sample,status,distance,gx,gy,gz,initial_sign\n";
                    std::array<std::uint64_t,4> counts{};
                    std::uint64_t batches=0,points=0,renewalSegments=0,negativeStarts=0;
                    double entryDistance=r.entryDistance,maximumDistance=r.length;
                    for (int i=0; i<trials; ++i) {
                        Random event(firstHitStreamSeed(seed,r.pixel,i,1));
                        Random gradient(firstHitStreamSeed(seed,r.pixel,i,2));
                        int status=0,initialSign=1;
                        double distance=r.length;
                        Vector3 g=Vector3::Zero();
                        if (m.family == "classic_local") {
                            const auto h = local.sample(localFlight,event,config.numeric);
                            distance=h.age;
                            if (h.collided) { status=1; g=sampleCollisionGradient(localFlight,h.age,gradient,config.numeric); }
                        } else if (m.family == "global_conditional") {
                            const auto birth = conditional.startExternal(r.entry,r.ray.direction,event);
                            const auto flight = conditional.beginFlight(birth);
                            const auto h = conditional.sample(flight,event,nullptr,r.length);
                            distance=h.age;
                            if (h.collided) { status=1; g=sampleCollisionGradient(flight,h.age,event,config.numeric); }
                        } else if (m.family == "global_conditional_renewal_plus") {
                            const auto birth = renewal.startExternal(r.entry,r.ray.direction,event);
                            const auto h = renewal.sample(birth,event,nullptr,r.length,&renewalSegments,config.numeric);
                            distance=h.age;
                            if (h.collided) { status=1; g=h.gradient; }
                        } else {
                            TungstenRandom rng(event.integer());
                            const auto h = gpis.sample(r.ray,std::nullopt,rng);
                            if (!h.enteredDomain) throw std::runtime_error("GPIS disagrees on domain intersection");
                            switch (h.status) {
                                case GpisFlightStatus::Escape: status=0; break;
                                case GpisFlightStatus::Hit: status=1; break;
                                case GpisFlightStatus::RejectedUpcrossing: status=2; break;
                                case GpisFlightStatus::InvalidGradient: status=3; break;
                            }
                            distance=h.distance; g=h.surface.gradient; initialSign=h.initialSign;
                            batches+=h.batches; points+=h.points;
                            entryDistance=h.entryDistance; maximumDistance=h.maximumDistance;
                        }
                        if (!std::isfinite(distance) || distance < 0 || distance > maximumDistance+1e-6)
                            throw std::runtime_error("first-flight distance outside interval");
                        if (status == 1 && (!g.allFinite() || !(g.squaredNorm()>0) || g.dot(r.ray.direction)>1e-6))
                            throw std::runtime_error("invalid accepted downcrossing gradient");
                        ++counts[status]; negativeStarts += initialSign<0;
                        samples << i << ',' << status << ',' << distance << ',' << g.x() << ','
                                << g.y() << ',' << g.z() << ',' << initialSign << '\n';
                    }
                    samples.close();
                    if (!samples) throw std::runtime_error("sample write failed");
                    const double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
                    results[job]={{"ray",id},{"method",m.id},{"trials",trials},{"escapes",counts[0]},
                        {"hits",counts[1]},{"rejected_upcrossings",counts[2]},{"invalid_gradients",counts[3]},
                        {"negative_initial_sign",negativeStarts},{"entry_distance",entryDistance},
                        {"maximum_distance",maximumDistance},{"gpis_batches",batches},{"gpis_points",points},
                        {"renewal_segments",renewalSegments},{"seconds_including_csv",seconds},{"file",filename}};
                    std::lock_guard<std::mutex> guard(lock);
                    std::cout << "ray " << id << " " << m.id << ": hits=" << counts[1]
                        << " escape=" << counts[0] << " rejected=" << counts[2]
                        << " invalid=" << counts[3] << " seconds=" << seconds << std::endl;
                    writeJson(output/(filename+".json"),results[job]);
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> guard(lock);
                if (!failure) failure=std::current_exception();
            }
        };
        std::vector<std::thread> workers;
        for (int i=0; i<std::min<int>(threads,static_cast<int>(count)); ++i) workers.emplace_back(worker);
        for (auto& thread : workers) thread.join();
        if (failure) std::rethrow_exception(failure);
        summary["samples"]=results;
        summary["preparation_seconds"]=preparationSeconds;
        summary["sampling_wall_seconds"]=std::chrono::duration<double>(Clock::now()-sampleStart).count();
        summary["total_seconds"]=std::chrono::duration<double>(Clock::now()-start).count();
        writeJson(output/"summary.json",summary);
        std::cout << (output/"summary.json").string() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

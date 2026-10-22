// Prepare single-scene ray splits and run the existing full-covariance reference.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/core/Random.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace {
using namespace mf;
using Json = nlohmann::json;
Json read(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    Json value; stream >> value; return value;
}
void write(const std::filesystem::path& path, const Json& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write " + path.string());
    stream << std::setprecision(17) << value.dump(2) << '\n';
}
Json vector(const Vector3& v) { return {v.x(),v.y(),v.z()}; }
Vector3 vector(const Json& v) {
    return Vector3(v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>());
}

std::optional<Point3> surfacePoint(const ExperimentConfig& scene, const Ray& ray, double end) {
    const double step = scene.field.mean->voxelSizeHint() * .5;
    double previous = 0, value = scene.field.mean->evaluate(ray.origin).value;
    for (double s = std::min(step,end); s <= end;) {
        const double next = scene.field.mean->evaluate(ray.origin+s*ray.direction).value;
        if (value > 0 && next <= 0) {
            double lo = previous, hi = s;
            for (int i=0;i<40;++i) {
                const double middle = (lo+hi)*.5;
                if (scene.field.mean->evaluate(ray.origin+middle*ray.direction).value > 0) lo=middle;
                else hi=middle;
            }
            return ray.origin+((lo+hi)*.5)*ray.direction;
        }
        if (s == end) break;
        previous=s; value=next; s=std::min(s+step,end);
    }
    return std::nullopt;
}

void prepare(const std::filesystem::path& specification, const std::filesystem::path& output) {
    const auto spec=read(specification);
    auto scene=loadExperimentConfig(spec.at("render_config").get<std::string>());
    prepareNanoVdbField(scene);
    const double sigma=scene.field.kernel.sigma();
    if (scene.field.kernel.type()!=CovarianceKernelType::SquaredExponential ||
        !scene.materialRoughness || std::abs(*scene.materialRoughness-spec.at("roughness").get<double>())>1e-12 ||
        static_cast<float>(sigma)!=static_cast<float>(spec.at("sigma").get<double>()))
        throw std::invalid_argument("collection scene does not match the requested SE sigma/roughness");
    const double ell=1/std::sqrt(scene.field.kernel.metric()(0,0));
    if (!scene.field.kernel.metric().isApprox(Matrix3::Identity()/(ell*ell),1e-10))
        throw std::invalid_argument("collection requires an isotropic SE kernel");
    const auto target=scene.render.cameraTarget;
    const Vector3 offset=scene.render.cameraPosition-target;
    const double radius=offset.norm(), azimuth=std::atan2(offset.y(),offset.x());
    Json records=Json::array(), cameras=Json::array();
    int splitIndex=0;
    for (const std::string split : {"train","validation","test"}) {
        const auto& settings=spec.at("splits").at(split);
        const int count=settings.at("profiles_per_mode"), views=settings.at("cameras");
        if (count<views || views<1) throw std::invalid_argument("invalid camera/profile counts");
        for (int view=0;view<views;++view) {
            const double angle=azimuth+2*kPi*(view+settings.at("azimuth_phase").get<double>())/views;
            const auto elevations=settings.at("elevations_degrees");
            const double elevation=elevations.at(view%elevations.size()).get<double>()*kPi/180;
            const Point3 position=target+radius*Vector3(std::cos(elevation)*std::cos(angle),
                std::cos(elevation)*std::sin(angle),std::sin(elevation));
            const auto forward=normalizedOrThrow(target-position);
            Vector3 right,up; orthonormalComplement(forward,right,up);
            const double scale=std::tan(scene.render.verticalFovDegrees*kPi/360);
            const double aspect=double(scene.render.width)/scene.render.height;
            const int needed=count/views+(view<count%views ? 1 : 0);
            const std::string camera=split+"_"+std::to_string(view);
            cameras.push_back({{"id",camera},{"split",split},{"position",vector(position)},
                               {"target",vector(target)},{"profiles_per_mode",needed}});
            for (int mode=0;mode<2;++mode) {
                Random rng(spec.at("seed").get<std::uint64_t>()+splitIndex*10000019ULL+view*104729ULL+mode*1000000007ULL);
                std::set<std::pair<int,int>> pixels;
                int accepted=0;
                const int strata=static_cast<int>(std::ceil(std::sqrt(double(needed))));
                for (int attempt=0;accepted<needed && attempt<50000;++attempt) {
                    // Initially spread candidates over the image; subsequent trials fill misses.
                    const int cell=attempt%(strata*strata);
                    const double sx=(cell%strata+rng.openUniform01())/strata;
                    const double sy=(cell/strata+rng.openUniform01())/strata;
                    const int px=std::min(scene.render.width-1,int(sx*scene.render.width));
                    const int py=std::min(scene.render.height-1,int(sy*scene.render.height));
                    if (pixels.count({px,py})) continue;
                    const double u=(2*sx-1)*aspect*scale, v=(1-2*sy)*scale;
                    Ray ray{position,normalizedOrThrow(forward+u*right+v*up)};
                    auto domain=scene.field.activeDomain.intersect(ray);
                    if (!domain.hit || domain.exit<=domain.entry) continue;
                    ray.origin=(ray.origin+domain.entry*ray.direction)
                        .cwiseMax(scene.field.activeDomain.minimum).cwiseMin(scene.field.activeDomain.maximum);
                    const double cameraLength=(domain.exit-domain.entry)*(1-1e-10);
                    Json extra=Json::object();
                    if (mode==1) {
                        const auto point=surfacePoint(scene,ray,cameraLength);
                        if (!point) continue;
                        const auto normal=normalizedOrThrow(scene.field.mean->evaluate(*point).gradient);
                        // Cover plausible zero observations on both sides of the mean surface.
                        const double displacement=(4*rng.openUniform01()-2)*sigma;
                        ray.origin=*point+displacement*normal;
                        if (!scene.field.activeDomain.contains(ray.origin)) continue;
                        Vector3 tangent,bitangent; orthonormalComplement(normal,tangent,bitangent);
                        const double cosine=.002+.998*rng.openUniform01();
                        const double phi=2*kPi*rng.openUniform01();
                        ray.direction=normalizedOrThrow(cosine*normal+std::sqrt(1-cosine*cosine)*
                            (std::cos(phi)*tangent+std::sin(phi)*bitangent));
                        extra={{"mean_surface_position",vector(*point)},{"mean_surface_normal",vector(normal)},
                               {"birth_offset_in_sigma",displacement/sigma},{"outgoing_cosine",cosine}};
                    }
                    domain=scene.field.activeDomain.intersect(ray);
                    const double length=domain.exit*(1-1e-10);
                    if (!domain.hit || domain.entry>1e-10 || length/ell<.25) continue;
                    scene.field.mean->requireFullRayCoverage(ray.origin,ray.direction,length);
                    const auto profile=RayMeanProfile::fromField(*scene.field.mean,ray.origin,ray.direction,
                        sigma,ell,length,spec.at("profile_maximum_step"));
                    const auto& first=profile.segments().front();
                    if (mode==1 && spec.contains("surface_residual_limit") &&
                        first.derivative(0)<=-spec.at("surface_residual_limit").get<double>()) continue;
                    if (mode==1 && spec.contains("surface_residual_limit") &&
                        first.derivative(0)<=-spec.at("surface_residual_limit").get<double>()) continue;
                    records.push_back({{"id",camera+"_"+std::to_string(mode)+"_"+std::to_string(accepted)},
                        {"split",split},{"camera_id",camera},{"pixel",{px,py}},
                        {"mode",mode==0 ? "positive_exterior" : "collision_state"},
                        {"origin",vector(ray.origin)},{"direction",vector(ray.direction)},
                        {"horizon",length/ell},{"physical_length",length},
                        {"entry_mean_sigma",first.value(0)},{"entry_mean_derivative",first.derivative(0)},
                        {"profile_segments",profile.segments().size()},{"geometry_selection",extra}});
                    pixels.insert({px,py}); ++accepted;
                }
                if (accepted!=needed) throw std::runtime_error("insufficient valid rays for "+camera);
            }
            std::cout << "prepared " << camera << " (" << needed << " rays per mode)" << std::endl;
        }
        ++splitIndex;
    }
    write(output,{{"sigma",sigma},{"roughness",*scene.materialRoughness},{"ell",ell},
        {"gradient_variance",scene.field.kernel.gradientCovarianceAtZero()(0,0)},
        {"domain_min",vector(scene.field.activeDomain.minimum)},{"domain_max",vector(scene.field.activeDomain.maximum)},
        {"cameras",cameras},{"profiles",records}});
}

void sample(const std::filesystem::path& planPath,int workers) {
    const auto plan=read(planPath);
    const auto& jobs=plan.at("jobs");
    if (jobs.empty() || workers<1) throw std::invalid_argument("empty sampling plan or invalid workers");
    const auto firstPath=jobs.at(0).at("input").get<std::string>();
    const auto base=loadFirstPassageExperimentConfig(firstPath);
    const auto fixed=read(firstPath).at("first_passage");
    std::atomic<std::size_t> next{0},completed{0};
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::string error;
    std::vector<std::thread> threads;
    for (int worker=0;worker<workers;++worker) threads.emplace_back([&] {
        while (!failed) {
            const auto i=next.fetch_add(1);
            if (i>=jobs.size()) break;
            try {
                const auto& job=jobs.at(i);
                const std::filesystem::path stamp=job.at("stamp").get<std::string>();
                bool cached=std::filesystem::exists(stamp) && read(stamp)==job.at("fingerprint");
                for (const std::string name : {"first_passage_samples.csv","first_passage_mean_segments.csv",
                    "first_passage_summary.csv","resolved_first_passage_config.json"})
                    cached=cached && std::filesystem::exists(stamp.parent_path()/name);
                if (!cached) {
                    const auto root=read(job.at("input").get<std::string>());
                    const auto& fp=root.at("first_passage");
                    for (const std::string key : {"process","kernels","sampler","profile","curve"})
                        if (fp.at(key)!=fixed.at(key)) throw std::invalid_argument("mixed reference definitions in plan");
                    auto config=base;
                    config.seed=root.at("seed");
                    config.outputDirectory=root.at("output_directory").get<std::string>();
                    config.initialConditionType=fp.at("initial_condition").at("type").get<std::string>();
                    if (config.initialConditionType!="positive_exterior" && config.initialConditionType!="collision_state")
                        throw std::invalid_argument("collection supports only Renewal A/B starts");
                    config.maximumTime=fp.at("grid").at("max_time");
                    config.stepSizes=fp.at("grid").at("step_sizes").get<std::vector<double>>();
                    config.trajectories=fp.at("monte_carlo").at("trajectories");
                    config.threadCount=1;
                    config.collisionStates.clear();
                    for (const auto& r : fp.at("initial_condition").at("rays")) {
                        FirstPassageCollisionState state;
                        state.id=r.at("id").get<std::string>();
                        FirstPassageRayCondition ray;
                        ray.origin=vector(r.at("origin")); ray.direction=normalizedOrThrow(vector(r.at("direction")));
                        if (config.initialConditionType=="collision_state") ray.gradient=vector(r.at("gradient"));
                        state.ray=ray; config.collisionStates.push_back(state);
                    }
                    runFirstPassageExperiment(config);
                    write(stamp,job.at("fingerprint"));
                }
                const auto count=completed.fetch_add(1)+1;
                std::lock_guard<std::mutex> lock(mutex);
                std::cout << "completed " << count << '/' << jobs.size() << " " << job.at("id").get<std::string>()
                          << (cached ? " cached" : "") << std::endl;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failed.exchange(true)) error=e.what();
            }
        }
    });
    for (auto& thread:threads) thread.join();
    if (failed) throw std::runtime_error(error);
}
}
int main(int argc,char** argv) {
    try {
        mf::registerNanoVdbFieldTypes();
        if (argc==4 && std::string(argv[1])=="prepare") prepare(argv[2],argv[3]);
        else if (argc==4 && std::string(argv[1])=="sample") sample(argv[2],std::stoi(argv[3]));
        else throw std::invalid_argument("usage: macrofacet_collect_shader_ball_dataset prepare <spec.json> <geometry.json> | sample <jobs.json> <workers>");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

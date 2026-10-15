// Fixed-camera-ray joint sampling through the same adapter as hybrid rendering.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/transport/FirstHitSEMedium.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    using namespace mf;
    using Json = nlohmann::json;
    if (argc != 4) {
        std::cerr << "usage: macrofacet_compare_first_hit_render <RNO hybrid config> <output directory> <trials per ray>\n";
        return 2;
    }
    try {
        registerNanoVdbFieldTypes();
        auto config = loadExperimentConfig(argv[1]);
        if (config.transportMode != "hybrid_first_hit" || config.firstHit.method != "neural_first_hit_se")
            throw std::invalid_argument("comparison needs a neural first-hit hybrid config");
        prepareFirstHitModel(config);
        const int trials = std::stoi(argv[3]);
        if (trials < 1 || trials > 1000000) throw std::invalid_argument("trials must be 1..1000000");
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        const FirstHitSEMedium neural(config.field,config.firstHit.model,config.firstHit.profileMaximumStep,config.firstHit.reference);
        const FirstHitSEMedium reference(config.field,nullptr,config.firstHit.profileMaximumStep,config.firstHit.reference);
        const auto forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
        Vector3 right, up; orthonormalComplement(forward,right,up);
        const double scale = std::tan(config.render.verticalFovDegrees*kPi/360.0);
        const double aspect = double(config.render.width)/config.render.height;
        const std::array<std::array<int,2>,3> pixels{{
            {config.render.width/2,config.render.height/2},
            {config.render.width*9/32,config.render.height*23/32},
            {config.render.width*23/32,config.render.height*23/32}}};
        Json rays = Json::array();
        std::vector<std::future<Json>> jobs;
        const auto vectorJson = [](const Vector3& v) { return Json::array({v.x(),v.y(),v.z()}); };
        for (std::size_t id = 0; id < pixels.size(); ++id) {
            const auto px = pixels[id][0], py = pixels[id][1];
            const auto pixel = std::uint64_t(py)*config.render.width+px;
            Random cameraRng(firstHitStreamSeed(config.seed,pixel,0,0));
            const double u = (2*((px+cameraRng.openUniform01())/config.render.width)-1)*aspect*scale;
            const double v = (1-2*((py+cameraRng.openUniform01())/config.render.height))*scale;
            const Ray ray{config.render.cameraPosition,normalizedOrThrow(forward+u*right+v*up)};
            const auto flight = neural.beginFlight(ray);
            if (!flight) throw std::runtime_error("selected comparison ray misses the domain");
            rays.push_back({{"id",id},{"pixel",pixels[id]},{"direction",vectorJson(ray.direction)},
                {"entry",vectorJson(flight->ray.origin)},{"entry_distance",flight->entryDistance},
                {"ell",flight->ell},{"sigma",flight->sigma},{"maximum_distance",flight->mean.maximumX()*flight->ell}});
            std::vector<double> q;
            for (int j = 0; j <= 256; ++j) q.push_back(flight->mean.maximumX()*j/256.0);
            const auto predictions = config.firstHit.model->query(flight->mean,q);
            std::ofstream curves(output/("ray_"+std::to_string(id)+"_curve.csv"));
            curves << std::setprecision(17) << "distance,transmittance\n";
            for (const auto& p : predictions) curves << p.q*flight->ell << ',' << std::exp(p.logTransmittance) << '\n';
            for (int method = 0; method < 2; ++method) {
                jobs.push_back(std::async(std::launch::async,[&,flight,id,pixel,method] {
                    const auto start = std::chrono::steady_clock::now();
                    const std::string label = method == 0 ? "rno" : "gpis";
                    const auto& medium = method == 0 ? neural : reference;
                    std::ofstream samples(output/("ray_"+std::to_string(id)+"_"+label+".csv"));
                    if (!samples) throw std::runtime_error("cannot write fixed-ray samples");
                    samples << std::setprecision(17) << "sample,hit,initial_sign,q,distance,speed,gx,gy,gz,nx,ny,nz\n";
                    int hits = 0, up = 0;
                    for (int i = 0; i < trials; ++i) {
                        Random event(firstHitStreamSeed(config.seed,pixel,i,1));
                        Random gradient(firstHitStreamSeed(config.seed,pixel,i,2));
                        const auto h = medium.sample(*flight,event,gradient,config.numeric.distanceAbsoluteTolerance);
                        hits += h.hit; up += h.hit && h.initialSign < 0;
                        samples << i << ',' << h.hit << ',' << h.initialSign << ',' << h.q << ',' << h.distance << ',' << h.speed;
                        for (const Vector3& vec : {h.gradient,h.normal})
                            samples << ',' << vec.x() << ',' << vec.y() << ',' << vec.z();
                        samples << '\n';
                    }
                    return Json{{"ray",id},{"method",label},{"trials",trials},{"hits",hits},{"upcrossings",up},
                        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
                }));
            }
        }
        Json summary{{"source_config",argv[1]},{"rays",rays},{"samples",Json::array()},
            {"gradient_reference","longitudinal Tungsten secant plus analytic conditional SE transverse reconstruction"}};
        for (auto& job : jobs) summary["samples"].push_back(job.get());
        std::ofstream summaryFile(output/"summary.json");
        summaryFile << std::setw(2) << summary << '\n';
        std::cout << (output/"summary.json").string() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

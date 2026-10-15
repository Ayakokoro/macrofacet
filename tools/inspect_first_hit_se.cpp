// Camera-entry geometry, unconditional SE reference requests, and scalar RNO/GRU queries.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/learned/FirstHitSEModel.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using Json=nlohmann::json;
Json vectorJson(const mf::Vector3& v) { return {v.x(),v.y(),v.z()}; }
void write(const std::filesystem::path& path,const Json& value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write "+path.string());
    stream << std::setw(2) << value << '\n';
}
}

int main(int argc,char** argv) {
    if (argc<5 || argc>7) {
        std::cerr << "usage: macrofacet_inspect_first_hit_se <render-config> <output-directory> <pixel-x> <pixel-y> [first-hit-model.json [trials]]\n";
        return 2;
    }
    try {
        using namespace mf;
        std::ifstream input(argv[1]);
        Json source; input >> source;
        if (source.at("field").value("mean_type","")!="nanovdb")
            throw std::invalid_argument("camera collection currently requires an imported full-domain NanoVDB field");
        registerNanoVdbFieldTypes();
        auto config=loadExperimentConfig(argv[1]);
        prepareNanoVdbField(config);
        if (config.field.kernel.type()!=CovarianceKernelType::SquaredExponential)
            throw std::invalid_argument("first-hit collection requires a stationary SE kernel");
        const int px=std::stoi(argv[3]),py=std::stoi(argv[4]);
        if (px<0 || px>=config.render.width || py<0 || py>=config.render.height)
            throw std::invalid_argument("pixel outside image");
        const auto pixel=static_cast<std::uint64_t>(py)*config.render.width+px;
        Random rng(config.seed+0x9e3779b97f4a7c15ULL*(pixel+1));
        const auto forward=normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
        Vector3 right,up; orthonormalComplement(forward,right,up);
        const double aspect=static_cast<double>(config.render.width)/config.render.height;
        const double scale=std::tan(config.render.verticalFovDegrees*kPi/360.);
        const double jitterX=rng.openUniform01(),jitterY=rng.openUniform01();
        const double u=(2*((px+jitterX)/config.render.width)-1)*aspect*scale;
        const double v=(1-2*((py+jitterY)/config.render.height))*scale;
        Ray ray{config.render.cameraPosition,normalizedOrThrow(forward+u*right+v*up)};
        const auto domain=config.field.activeDomain.intersect(ray);
        if (!domain.hit || !(domain.exit>domain.entry)) throw std::invalid_argument("ray misses medium");
        ray.origin+=domain.entry*ray.direction;
        ray.origin=ray.origin.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
        const double ell=1/std::sqrt(ray.direction.dot(config.field.kernel.metric()*ray.direction));
        const double sigma=config.field.kernel.sigma();
        const double maximum=(domain.exit-domain.entry)*(1-1e-12), Q=maximum/ell;
        const auto mean=RayMeanProfile::fromField(*config.field.mean,ray.origin,ray.direction,sigma,ell,maximum,.25);
        Json segments=Json::array(), covariance=Json::array();
        for (const auto& segment:mean.segments()) segments.push_back(segment.features());
        const auto C=config.field.kernel.gradientCovarianceAtZero();
        for (int i=0;i<3;++i) covariance.push_back({C(i,0),C(i,1),C(i,2)});
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        write(output/"mean_profile.json",{{"features",segments},{"horizon",Q}});
        write(output/"ray.json",{{"event","unconditioned_first_zero"},{"source_config",argv[1]},
            {"pixel",{px,py}},{"entry_distance",domain.entry},{"entry_origin",vectorJson(ray.origin)},
            {"direction",vectorJson(ray.direction)},{"sigma",sigma},{"ell",ell},{"horizon",Q},
            {"gradient_covariance",covariance},{"distance_convention","distance from activeDomain entry"}});
        const double referenceStep=Q/std::ceil(Q*64);
        Json reference{{"schema_version",1},{"seed",config.seed+20261013},
            {"first_passage",{
                {"process",{{"mean",0},{"threshold",0},{"mean_field",source.at("field")}}},
                {"initial_condition",{{"type","unconditioned_first_zero"},{"rays",Json::array({
                    {{"id","camera_pixel"},{"origin",vectorJson(ray.origin)},{"direction",vectorJson(ray.direction)}}})}}},
                {"grid",{{"max_time",Q},{"step_sizes",{referenceStep,referenceStep/2}}}},
                {"profile",{{"maximum_step",.25}}},{"curve",{{"bins",256}}},
                {"monte_carlo",{{"trajectories",4096},{"thread_count",4},{"write_raw_samples",true}}},
                {"rice_series",{{"enabled",false}}},{"state_analysis",{{"enabled",false}}},
                {"kernels",Json::array({{{"id","se"},{"type","squared_exponential"},{"parameterization","unit_length"},
                    {"variance",sigma*sigma},{"length_scale",ell}}})}}},
            {"output_directory",(output/"reference").string()}};
        write(output/"reference.json",reference);
        if (argc>=6) {
            const auto model=FirstHitSEModel::load(argv[5]);
            std::vector<double> queries;
            for (int i=0;i<=256;++i) queries.push_back(Q*i/256.);
            const auto predictions=model.query(mean,queries);
            std::ofstream curves(output/"neural_curve.csv");
            curves << std::setprecision(17) << "q,distance_from_entry,transmittance,H_positive,H_negative,positive_hit_probability\n";
            Json queryJson=Json::array();
            for (const auto& p:predictions) {
                curves << p.q << ',' << p.q*ell << ',' << std::exp(p.logTransmittance) << ','
                    << p.cumulative[0] << ',' << p.cumulative[1] << ',' << p.hitBranchProbability[0] << '\n';
                Json mixtures=Json::array();
                for (const auto& mixture:p.mixture) mixtures.push_back({{"weights",mixture.weights},{"means",mixture.means},{"scales",mixture.scales}});
                queryJson.push_back({{"q",p.q},{"log_T",p.logTransmittance},{"H",p.cumulative},
                    {"log_hazard",p.logHazard},{"hit_branch_probability",p.hitBranchProbability},{"mixtures",mixtures}});
            }
            write(output/"predictions.json",queryJson);
            const int trials=argc==7 ? std::stoi(argv[6]) : 512;
            if (trials<0) throw std::invalid_argument("trials cannot be negative");
            std::ofstream samples(output/"samples.csv");
            samples << std::setprecision(17) << "sample,hit,initial_sign,q,distance_from_entry,speed,directional_derivative,gx,gy,gz,nx,ny,nz\n";
            std::size_t hits=0,upcrossings=0;
            for (int i=0;i<trials;++i) {
                const auto hit=model.sample(mean,rng);
                samples << i << ',' << hit.hit << ',' << hit.initialSign << ',' << hit.q << ',' << hit.q*ell;
                if (hit.hit) {
                    ++hits; if (hit.initialSign<0) ++upcrossings;
                    std::size_t j=0;
                    while (j+1<mean.segments().size() && hit.q>mean.segments()[j].end) ++j;
                    const auto& segment=mean.segments()[j];
                    const Point3 point=ray.origin+hit.q*ell*ray.direction;
                    const Point3 interior=ray.origin+(segment.begin+segment.end)*.5*ell*ray.direction;
                    const auto mu=config.field.mean->evaluateInCell(point,interior).gradient;
                    const double derivative=-hit.initialSign*sigma/ell*hit.speed;
                    const auto gradient=sampleFirstHitSEGradient(config.field.kernel,ray.direction,derivative,mu,rng);
                    const auto normal=normalizedOrThrow(gradient);
                    samples << ',' << hit.speed << ',' << derivative << ',' << gradient.x() << ',' << gradient.y() << ',' << gradient.z()
                        << ',' << normal.x() << ',' << normal.y() << ',' << normal.z();
                } else samples << ",,,,,,,,";
                samples << '\n';
            }
            write(output/"sample_summary.json",{{"architecture",model.architecture()},{"trials",trials},{"hits",hits},
                {"upcrossings",upcrossings},{"predicted_terminal_T",std::exp(predictions.back().logTransmittance)}});
        }
        std::cout << (output/"reference.json").string() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

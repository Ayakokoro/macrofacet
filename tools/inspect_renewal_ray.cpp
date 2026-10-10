// Inspect the first camera sample of one pixel using the renderer's field,
// start condition, segmentation and backend. Reference sampling remains in
// the existing first-passage command, separate from neural inference.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
Json vectorJson(const mf::Vector3& v) { return Json::array({v.x(),v.y(),v.z()}); }
void writeJson(const std::filesystem::path& path,const Json& value) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write "+path.string());
    file << value.dump(2) << '\n';
}
}

int main(int argc,char** argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "usage: macrofacet_inspect_renewal_ray <render-config> <output-directory> [pixel-x pixel-y]\n";
        return 2;
    }
    try {
        using namespace mf;
        std::ifstream input(argv[1]);
        Json source; input >> source;
        if (source.at("field").value("mean_type","") != "nanovdb")
            throw std::invalid_argument("ray inspection currently requires an imported full-domain NanoVDB render config");
        registerNanoVdbFieldTypes();
        auto config = loadExperimentConfig(argv[1]);
        if (config.transportMode != "neural_renewal")
            throw std::invalid_argument("ray inspection requires neural_renewal mode");
        prepareRenewalModel(config);
        prepareNanoVdbField(config);
        const int px = argc == 5 ? std::stoi(argv[3]) : config.render.width/2;
        const int py = argc == 5 ? std::stoi(argv[4]) : config.render.height/2;
        if (px < 0 || px >= config.render.width || py < 0 || py >= config.render.height)
            throw std::invalid_argument("pixel is outside the render image");
        const std::uint64_t pixel = static_cast<std::uint64_t>(py)*config.render.width+px;
        const auto pixelSeed = config.seed+0x9e3779b97f4a7c15ULL*(pixel+1);
        Random rng(pixelSeed);
        const auto forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
        Vector3 right,up; orthonormalComplement(forward,right,up);
        const double aspect = static_cast<double>(config.render.width)/config.render.height;
        const double scale = std::tan(config.render.verticalFovDegrees*kPi/360.0);
        const double jitterX = rng.openUniform01(), jitterY = rng.openUniform01();
        const double u = (2.0*((px+jitterX)/config.render.width)-1.0)*aspect*scale;
        const double v = (1.0-2.0*((py+jitterY)/config.render.height))*scale;
        const Vector3 direction = normalizedOrThrow(forward+u*right+v*up);
        Ray ray{config.render.cameraPosition,normalizedOrThrow(direction)};
        const auto domain = config.field.activeDomain.intersect(ray);
        if (!domain.hit || !(domain.exit > domain.entry))
            throw std::invalid_argument("selected camera ray misses the active domain");
        ray.origin += domain.entry*ray.direction;
        ray.origin = ray.origin.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
        const auto& model = *config.renewal.model;
        const bool se = model.kernelType() == CovarianceKernelType::SquaredExponential;
        const std::string kernelType = covarianceKernelTypeName(model.kernelType());
        const std::string parameterization = se ? "unit_length" : "unit_decay";
        const RenewalMedium medium(config.field,model,config.renewal.profileMaximumStep,config.renewal.profileMode);
        const auto flight = medium.beginFlight(ray);
        if (!flight) throw std::runtime_error("selected camera ray has no flight inside the domain");
        // Query cutoffs never alter the renderer's original neural segments.
        // Leave only a roundoff-sized margin for the independent field preflight.
        const double maxX = flight->mean.maximumX()*(1.0-1e-12);
        const double coarseStep = maxX/std::ceil(maxX*32.0);
        const int bins = 4096;
        const double opticalDepth = -std::log(rng.openUniform01());
        std::optional<RenewalBatchSession> batch;
        if (config.renewal.resolvedBackend != "scalar")
            batch.emplace(model,1,config.renewal.resolvedBackend);
        auto state = model.initialize(flight->mean.segment(0),flight->start);
        if (batch) batch->initialize({0},{flight->mean.segment(0)},{flight->start});
        std::vector<RenewalHazardSegment> hazards,scalarHazards;
        Json segments = Json::array();
        for (std::size_t i = 0; flight->mean.hasSegment(i); ++i) {
            const auto& segment = flight->mean.segment(i);
            auto scalar = model.evaluate(state,segment);
            const auto rates = batch ? batch->evaluate({0},{segment}).front() : scalar.rates;
            hazards.emplace_back(segment.begin,segment.end,rates);
            scalarHazards.emplace_back(segment.begin,segment.end,scalar.rates);
            state = std::move(scalar.nextState);
            segments.push_back({{"begin_x",segment.begin},{"end_x",segment.end},
                                {"features",segment.features()},{"rates",rates}});
        }
        const RenewalRayDistribution distribution(std::move(hazards),flight->ell);
        const RenewalRayDistribution scalarDistribution(std::move(scalarHazards),flight->ell);
        const auto sampled = distribution.sampleOpticalDepth(opticalDepth,0,flight->maximumDistance(),
                                                           config.numeric.distanceAbsoluteTolerance);
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        std::ofstream curves(output/"neural_curve.csv");
        if (!curves) throw std::runtime_error("cannot write neural curve");
        curves << std::setprecision(17)
               << "x,distance_from_entry,distance_from_camera,transmittance,scalar_transmittance,cumulative_hazard,mean_sdf,profile_mean_sdf\n";
        std::size_t segmentIndex = 0;
        double maxBackendDifference = 0;
        double previous = 1;
        for (int i = 0; i <= bins; ++i) {
            const double x = maxX*i/bins, distance = x*flight->ell;
            while (segmentIndex+1 < segments.size() && x >= flight->mean.segment(segmentIndex).end) ++segmentIndex;
            const double t = distribution.transmittance(distance);
            const double ts = scalarDistribution.transmittance(distance);
            if (!std::isfinite(t) || t < 0 || t > previous+1e-12)
                throw std::runtime_error("invalid/nonmonotone neural transmittance");
            previous = t;
            maxBackendDifference = std::max(maxBackendDifference,std::abs(t-ts));
            curves << x << ',' << distance << ',' << domain.entry+distance << ',' << t << ',' << ts << ','
                   << distribution.cumulativeHazard(distance) << ','
                   << config.field.mean->evaluate(ray.origin+distance*ray.direction).value << ','
                   << flight->sigma*flight->mean.segment(segmentIndex).value(x) << '\n';
        }
        // This cross-check detects an unintended backend/model mismatch. It is
        // not an accuracy check against the GP reference.
        if (maxBackendDifference > 1e-4) throw std::runtime_error("large scalar/selected-backend transmittance difference");
        Json metadata{{"source_config",argv[1]},{"pixel",{px,py}},{"width",config.render.width},
            {"height",config.render.height},{"sample_index",0},{"pixel_seed",pixelSeed},
            {"jitter",{jitterX,jitterY}},{"camera_origin",vectorJson(config.render.cameraPosition)},
            {"entry_origin",vectorJson(ray.origin)},{"direction",vectorJson(ray.direction)},
            {"entry_distance",domain.entry},{"domain_exit_distance",domain.exit},
            {"reference_maximum_x",maxX},{"sigma",flight->sigma},{"ell",flight->ell},
            {"kernel",kernelType+"_"+parameterization},{"start_condition","positive_exterior"},
            {"profile_mode",config.renewal.profileMode},{"profile_maximum_step",config.renewal.profileMaximumStep},
            {"backend",config.renewal.resolvedBackend},{"checkpoint_sha256",model.checkpointSha256()},
            {"model",source.at("transport").at("renewal").at("model")},{"field",source.at("field")},
            {"neural_segments",segments.size()},{"max_scalar_backend_transmittance_difference",maxBackendDifference},
            {"replayed_sample_hit",sampled.hit},{"replayed_sample_distance_from_entry",sampled.distance},
            {"optical_depth",opticalDepth},
            {"note","Sample 0 camera ray reproduced from renderer RNG; selected backend replay uses batch size 1, so floating-point rounding may differ from full wavefront batches. Reference uses full trilinear mean; neural uses configured mean approximation."}};
        writeJson(output/"ray.json",metadata);
        writeJson(output/"neural_segments.json",segments);
        writeJson(output/"render_config_snapshot.json",source);
        for (int resolution = 0; resolution < 2; ++resolution) {
            const auto name = resolution ? "fine" : "coarse";
            const double step = std::ldexp(coarseStep,-resolution);
            Json reference{{"schema_version",1},{"seed",config.seed+20261009+resolution*104729},
                {"first_passage",{
                    {"process",{{"mean",0},{"threshold",0},{"mean_field",source.at("field")}}},
                    {"initial_condition",{{"type","positive_exterior"},{"rays",Json::array({
                        {{"id","camera_pixel"},{"origin",vectorJson(ray.origin)},{"direction",vectorJson(ray.direction)}}})}}},
                    {"grid",{{"max_time",maxX},{"step_sizes",{step}}}},
                    {"profile",{{"maximum_step",config.renewal.profileMaximumStep}}},
                    {"curve",{{"bins",bins}}},
                    {"monte_carlo",{{"trajectories",65536},{"thread_count",8},{"write_raw_samples",true}}},
                    {"sampler",{{"type","collision_state_auto"},{"minimum_step",step/64},
                        {"crossing_tolerance",1e-9},{"bridge_sigma_margin",6},{"max_refinement_depth",12}}},
                    {"rice_series",{{"enabled",false}}},{"state_analysis",{{"enabled",false}}},
                    {"kernels",Json::array({{{"id",se ? "se" : "matern32"},{"type",kernelType},
                        {"parameterization",parameterization},{"variance",flight->sigma*flight->sigma},
                        {"length_scale",flight->ell}}})}}},
                {"output_directory",(output/name).string()}};
            writeJson(output/(std::string("reference_")+name+".json"),reference);
        }
        std::cout << metadata.dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

// Fixed-position collision-gradient distributions using the production samplers.
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {
using Json = nlohmann::json;
using namespace mf;
Vector3 vector(const Json& v) { return {v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>()}; }
Json json(const Vector3& v) { return Json::array({v.x(),v.y(),v.z()}); }
Json read(const std::filesystem::path& p) { std::ifstream f(p); Json j; f >> j; return j; }
struct Sample { Vector3 gradient = Vector3::Zero(); double weight = 0.; };
}

int main(int argc, char** argv) {
    if (argc != 5 && !(argc == 7 && std::string(argv[5]) == "--distance")) {
        std::cerr << "usage: macrofacet_inspect_collision_gradients <render-config> <ray.json> <output> <trials> [--distance fixed-distance]\n";
        return 2;
    }
    try {
        const auto meta = read(argv[2]);
        const bool surface = meta.at("start_condition") == "collision_state";
        if ((!surface && meta.at("start_condition") != "positive_exterior") ||
            (argc == 5 && !meta.at("replayed_sample_hit").get<bool>()))
            throw std::invalid_argument("requires an A/B flight with a hit or an explicit fixed target distance");
        const int trials = std::stoi(argv[4]);
        if (trials < 1) throw std::invalid_argument("positive trial count required");
        registerNanoVdbFieldTypes();
        auto config = loadExperimentConfig(argv[1]);
        prepareRenewalModel(config);
        prepareNanoVdbField(config);
        const auto& model = *config.renewal.model;
        const auto& field = config.field;
        if (field.kernel.type() != CovarianceKernelType::SquaredExponential)
            throw std::invalid_argument("gradient comparison currently requires SE");
        const Ray ray{vector(meta.at("entry_origin")),normalizedOrThrow(vector(meta.at("direction")))};
        const double distance = argc == 7 ? std::stod(argv[6]) : meta.at("replayed_sample_distance_from_entry").get<double>();
        const std::optional<Vector3> birthGradient = surface
            ? std::optional<Vector3>(vector(meta.at("source_hit").at("gradient"))) : std::nullopt;
        RenewalMedium neural(field,model,config.renewal.profileMaximumStep,config.renewal.profileMode);
        const auto flight = neural.beginFlight(ray,birthGradient);
        if (!flight || !(distance > 0 && distance < flight->maximumDistance()))
            throw std::invalid_argument("invalid fixed target distance");
        if (std::abs(flight->sigma-meta.at("sigma").get<double>()) > 1e-12 ||
            std::abs(flight->ell-meta.at("ell").get<double>()) > 1e-12 ||
            model.checkpointSha256() != meta.at("checkpoint_sha256"))
            throw std::invalid_argument("ray metadata does not match field/model");
        const double q = distance / flight->ell;
        std::optional<RenewalBatchSession> batch;
        if (config.renewal.resolvedBackend != "scalar") batch.emplace(model,1,config.renewal.resolvedBackend);
        auto state = model.initialize(flight->mean.segment(0),flight->start);
        if (batch) batch->initialize({0},{flight->mean.segment(0)},{flight->start});
        std::size_t index = 0;
        RenewalSpeedMixture mixture;
        double coordinate = 0.;
        for (; flight->mean.hasSegment(index); ++index) {
            const auto& segment = flight->mean.segment(index);
            auto step = model.evaluate(state,segment);
            if (batch) batch->evaluate({0},{segment});
            if (q <= segment.end) {
                coordinate = (q-segment.begin)/(segment.end-segment.begin);
                mixture = batch ? batch->mixture({0},{coordinate}).front() : model.mixture(state,segment,coordinate);
                break;
            }
            state = std::move(step.nextState);
        }
        if (mixture.weights.empty()) throw std::runtime_error("target segment not found");
        auto material = config.material;
        material.gpModel = GpModel::LocalTangent;
        const ClassicFlightKernel classic(field,material,startExternalFlight(ray.origin,ray.direction));
        const auto localGaussian = classic.hitStatistics(distance).collisionGradient;
        const Point3 point = ray.origin+distance*ray.direction;
        const auto mean = field.mean->evaluate(point);
        const auto birthMean = field.mean->evaluate(ray.origin);
        Vector3 e1,e2;
        orthonormalComplement(ray.direction,e1,e2);
        const Matrix3 covariance = field.kernel.gradientCovarianceAtZero();
        const double transverseVariance = flight->sigma*flight->sigma/(flight->ell*flight->ell);
        if ((covariance-transverseVariance*Matrix3::Identity()).norm() > 1e-10)
            throw std::invalid_argument("GP transverse reconstruction currently requires isotropic SE");

        std::vector<Sample> locals(trials), networks(trials), conditionals(trials);
        const auto baseSeed = meta.at("pixel_seed").get<std::uint64_t>() ^ 0x6772616469656e74ULL ^
            (surface ? 0x7365636f6e646869ULL : 0ULL);
        Random localRng(baseSeed), neuralRng(baseSeed+104729);
        for (int i=0;i<trials;++i) {
            locals[i] = {sampleFluxWeightedGradient(localGaussian,ray.direction,localRng,config.numeric),1.};
            networks[i] = {neural.sampleHit(*flight,index,coordinate,mixture,neuralRng).gradient,1.};
        }
        // p(g | first hit at t) mixes birth observations with weight
        // p(B | F(entry)>0) T(t|B) hazard(t|B). A null-tracking survival
        // indicator is an unbiased T estimator, independent of the endpoint draw.
        const ConditionalMedium conditional(field);
        std::optional<Gaussian<3>> fixedBirthGaussian;
        if (surface) {
            auto birth = startExternalFlight(ray.origin,ray.direction);
            birth.birthValue = 0.;
            birth.birthGradient = *birthGradient;
            fixedBirthGaussian = conditional.beginFlight(birth).hitStatistics(distance).collisionGradient;
        }
        const unsigned workerCount = std::min(8u,static_cast<unsigned>(trials));
        std::vector<std::thread> workers;
        std::vector<std::exception_ptr> errors(workerCount);
        for (unsigned worker=0;worker<workerCount;++worker) workers.emplace_back([&,worker] {
            try {
                for (int i=static_cast<int>(worker);i<trials;i+=static_cast<int>(workerCount)) {
                    Random rng(baseSeed+20261011+0x9e3779b97f4a7c15ULL*static_cast<std::uint64_t>(i+1));
                    if (fixedBirthGaussian) {
                        // At fixed birth and target, survival*hazard is common
                        // to all endpoint gradients and cancels on normalization.
                        conditionals[i] = {sampleFluxWeightedGradient(*fixedBirthGaussian,ray.direction,rng,config.numeric),1.};
                        continue;
                    }
                    const auto birth = conditional.startExternal(ray.origin,ray.direction,rng);
                    const auto kernel = conditional.beginFlight(birth);
                    if (conditional.sample(kernel,rng,nullptr,distance).collided) continue;
                    const double weight = kernel.evaluate(distance).hazard.value;
                    if (!(weight >= 0) || !std::isfinite(weight)) throw std::runtime_error("invalid collision density weight");
                    if (weight > 0) conditionals[i] = {sampleCollisionGradient(kernel,distance,rng,config.numeric),weight};
                }
            } catch (...) { errors[worker]=std::current_exception(); }
        });
        for (auto& worker : workers) worker.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        const std::filesystem::path out(argv[3]);
        std::filesystem::create_directories(out);
        std::ofstream samples(out/"gradient_samples.csv");
        if (!samples) throw std::runtime_error("cannot write gradient samples");
        samples << std::setprecision(17) << "mode,trial,weight,gx,gy,gz,incoming_derivative,transverse_1,transverse_2\n";
        const auto emit = [&](const char* mode,const std::vector<Sample>& rows) {
            for (int i=0;i<trials;++i) {
                const auto& s=rows[i];
                if (s.weight>0 && !(ray.direction.dot(s.gradient)<0)) throw std::runtime_error("gradient violates downcrossing support");
                samples << mode << ',' << i << ',' << s.weight;
                for (int j=0;j<3;++j) samples << ',' << s.gradient[j];
                samples << ',' << -ray.direction.dot(s.gradient) << ',' << e1.dot(s.gradient) << ',' << e2.dot(s.gradient) << '\n';
            }
        };
        emit("classic_local",locals); emit("global_conditional",conditionals); emit("neural",networks);
        Json result{{"pixel",meta.at("pixel")},{"target_distance",distance},{"target_q",q},
            {"start_condition",meta.at("start_condition")},{"collision_index",surface ? 2 : 1},
            {"origin",json(ray.origin)},{"birth_mean_gradient",json(birthMean.gradient)},
            {"source_replayed_sample_hit",meta.at("replayed_sample_hit")},
            {"target_distance_from_camera",meta.at("entry_distance").get<double>()+distance},
            {"position",json(point)},{"direction",json(ray.direction)},{"e1",json(e1)},{"e2",json(e2)},
            {"sigma",flight->sigma},{"ell",flight->ell},{"mean_value",mean.value},{"mean_gradient",json(mean.gradient)},
            {"transverse_variance",transverseVariance},{"trials_per_mode",trials},
            {"neural_backend",config.renewal.resolvedBackend},{"neural_profile_mode",config.renewal.profileMode},
            {"neural_mixture",{{"weights",mixture.weights},{"means",mixture.means},{"scales",mixture.scales}}},
            {"conditioning","same fixed first-collision distance, F(target)=0, incoming gradient; each mode retains its own transport law"},
            {"global_conditional_weight","sample birth from F(entry)>0; survive to target with null tracking; weight by hazard(target|birth)"},
            {"gp_transverse_reconstruction","isotropic SE: transverse residual gradients are independent of all scalar observations along the ray; add independent N(0,sigma^2/ell^2) to mean transverse components"}};
        if (surface) {
            const double rho = std::exp(-0.5*q*q);
            result["source_hit"] = meta.at("source_hit");
            result["gp_transverse_mean"] = json(mean.gradient+rho*(*birthGradient-birthMean.gradient));
            result["gp_transverse_variance"] = transverseVariance*(-std::expm1(-q*q));
            result["conditional_directional_mean"] = ray.direction.dot(fixedBirthGaussian->mean);
            result["conditional_directional_variance"] = ray.direction.dot(fixedBirthGaussian->covariance*ray.direction);
            result["conditioning"] = "same fixed second collision position, conditioned on first-hit F=0 and full gradient; no earlier camera-path survival history; classic_local ignores the first-hit observation";
            result["global_conditional_weight"] = "fixed first-hit observation and fixed second target; survival*hazard cancels; unit-weight production flux-gradient samples";
            result["gp_transverse_reconstruction"] = "isotropic SE conditioned on first full gradient: mean=m_grad(target)+rho*(g_birth-m_grad(birth)), transverse variance=(sigma/ell)^2*(1-rho^2), rho=exp(-q^2/2)";
        }
        std::ofstream metadata(out/"target.json");
        metadata << result.dump(2) << '\n';
        std::cout << "Completed fixed collision gradient samples: " << out << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

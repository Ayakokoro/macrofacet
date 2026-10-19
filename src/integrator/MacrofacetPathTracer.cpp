#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include "macrofacet/integrator/RenewalWavefront.h"
#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include "macrofacet/learned/CausalFirstHitModel.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/transport/ConditionalMedium.h"
#include "macrofacet/transport/GlobalConditionalRenewalPlusMedium.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/transport/NarrowBandMedium.h"
#include "macrofacet/transport/FunctionSpaceGpisMedium.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <thread>

namespace mf {
namespace {

Spectrum environmentEmission(const Vector3& direction, const std::string& environment) {
    if (environment == "unit_white") return Spectrum::Ones();
    const double up = 0.5 * (normalizedOrThrow(direction).z() + 1.0);
    return Spectrum(0.15 + 0.85 * up, 0.2 + 0.7 * up, 0.35 + 0.55 * up);
}

int resolveWorkerCount(int requested, int rows) {
    if (requested > 0) return std::max(1, std::min(requested, rows));
    const unsigned hardware = std::thread::hardware_concurrency();
    const int available = hardware == 0 ? 1 : static_cast<int>(hardware);
    return std::max(1, std::min(available, rows));
}

Spectrum traceRenewalFlights(Ray ray, std::optional<Vector3> gradient,
    Spectrum throughput, int startDepth, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    for (int depth = startDepth; depth < config.render.safetyDepthCap; ++depth) {
        const auto flight = medium.beginFlight(ray, gradient);
        const auto hit = flight ? medium.sample(*flight, rng,
            config.numeric.distanceAbsoluteTolerance, &statistics.renewal) : RenewalSurfaceSample{};
        if (!hit.hit) {
            ++statistics.escapedPaths;
            statistics.accumulatedPathDepth += depth;
            return throughput.cwiseProduct(environmentEmission(ray.direction, config.render.environment));
        }
        ++statistics.realCollisions;
        throughput = throughput.cwiseProduct(conductorFresnel(-ray.direction.dot(hit.normal), config.material.conductor));
        const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(ray.direction, hit.normal));
        if (!(outgoing.dot(hit.gradient) > 0))
            throw std::runtime_error("Renewal reflection must leave the sampled surface");
        // The sampled geometry is part of the target model: no inverse hit PDF
        // or extra Rice factor. A new direction starts B with the FULL gradient.
        ray = {hit.position, outgoing};
        gradient = hit.gradient;
        if (depth+1 >= config.render.rouletteStartDepth) {
            const double probability = std::clamp(throughput.maxCoeff(), 0.05, 0.95);
            if (rng.openUniform01() >= probability) {
                ++statistics.rouletteTerminations;
                statistics.accumulatedPathDepth += depth+1;
                return Spectrum::Zero();
            }
            throughput /= probability;
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth += config.render.safetyDepthCap;
    return Spectrum::Zero();
}

Spectrum traceRenewalCameraPath(const Ray& input, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    Ray ray{input.origin, normalizedOrThrow(input.direction)};
    const auto interval = config.field.activeDomain.intersect(ray);
    if (!interval.hit || !(interval.exit > interval.entry)) {
        ++statistics.escapedPaths;
        return environmentEmission(ray.direction, config.render.environment);
    }
    ray.origin += interval.entry*ray.direction;
    ray.origin = ray.origin.cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
    return traceRenewalFlights(ray,std::nullopt,Spectrum::Ones(),0,config,medium,rng,statistics);
}

// Continue an existing path with the same classic/conditional kernel used by
// camera paths. startDepth counts reflections that have already been applied.
Spectrum traceTrackedFlights(FlightState state, Spectrum throughput, int startDepth,
    const ExperimentConfig& config, const NarrowBandMedium* classicMedium,
    const ConditionalMedium* conditionalMedium, Random& rng, RenderStatistics& statistics,
    const GlobalConditionalRenewalPlusMedium* conditionalRenewalMedium = nullptr) {
    const bool conditional = conditionalMedium != nullptr || conditionalRenewalMedium != nullptr;
    int depth = startDepth;
    for (; depth < config.render.safetyDepthCap; ++depth) {
        try {
            FlightSample flight;
            Vector3 collisionGradient = Vector3::Zero();
            if (conditionalRenewalMedium) {
                ++statistics.conditionalFlights;
                const auto sample = conditionalRenewalMedium->sample(state,rng,&statistics.tracking,
                    -1.0,&statistics.conditionalRenewalSegments,config.numeric);
                flight = {sample.collided,sample.age};
                collisionGradient = sample.gradient;
            } else if (conditional) {
                ++statistics.conditionalFlights;
                const ConditionalFlightKernel kernel = conditionalMedium->beginFlight(state);
                flight = conditionalMedium->sample(kernel, rng, &statistics.tracking);
                if (flight.collided)
                    collisionGradient = sampleCollisionGradient(kernel, flight.age,
                                                                 rng, config.numeric);
            } else {
                ++statistics.classicFlights;
                const ClassicFlightKernel kernel = classicMedium->beginFlight(state);
                flight = classicMedium->sample(kernel, rng, config.numeric, &statistics.tracking);
            }
            if (!flight.collided) {
                ++statistics.escapedPaths;
                statistics.accumulatedPathDepth += depth;
                return throughput.cwiseProduct(
                    environmentEmission(state.direction, config.render.environment));
            }
            const Point3 hit = state.birthPosition + flight.age * state.direction;
            ++statistics.realCollisions;
            if (conditional) {
                const Vector3 normal = normalizedOrThrow(collisionGradient);
                throughput = throughput.cwiseProduct(conductorFresnel(
                    -state.direction.dot(normal), config.material.conductor));
                const Vector3 outgoing = reflectTravelDirection(state.direction, normal);
                state = startClassicCollisionFlight(hit, outgoing);
                state.birthValue = 0.0;
                state.birthGradient = collisionGradient;
            } else {
                const double mixture = config.classicPhaseProposal == "paper_mixture"
                    ? config.beckmannMixtureWeight : 0.0;
                const bool useTargetVndf = config.classicPhaseProposal == "target_vndf";
                ConductorPhase phase(config.field, config.material, hit, mixture, useTargetVndf);
                const PhaseSample sample = phase.samplePhase(state.direction, rng);
                throughput = throughput.cwiseProduct(sample.throughputWeight);
                state = startClassicCollisionFlight(hit, sample.direction);
            }
            if (depth + 1 >= config.render.rouletteStartDepth) {
                const double continuation = std::clamp(throughput.maxCoeff(), 0.05, 0.95);
                if (rng.openUniform01() >= continuation) {
                    ++statistics.rouletteTerminations;
                    statistics.accumulatedPathDepth += depth + 1;
                    return Spectrum::Zero();
                }
                throughput /= continuation;
            }
        } catch (const NumericError&) {
            if (conditional) throw; // A failed majorant must not become a black sample.
            ++statistics.numericalFailures;
            statistics.accumulatedPathDepth += depth;
            return Spectrum::Zero();
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth += depth;
    return Spectrum::Zero();
}

Spectrum traceConditionalCameraPath(const Ray& ray, const ExperimentConfig& config,
    const ConditionalMedium* medium, Random& rng, RenderStatistics& statistics,
    const GlobalConditionalRenewalPlusMedium* renewalMedium = nullptr) {
    const Vector3 direction=normalizedOrThrow(ray.direction);
    const auto interval=config.field.activeDomain.intersect({ray.origin,direction});
    if (!interval.hit || !(interval.exit > interval.entry)) {
        ++statistics.escapedPaths;
        return environmentEmission(direction,config.render.environment);
    }
    const Point3 entry=(ray.origin+interval.entry*direction)
        .cwiseMax(config.field.activeDomain.minimum).cwiseMin(config.field.activeDomain.maximum);
    // Both conditional modes must use the same clipped world-space observation;
    // even roundoff at a grid face can change the subsequent tracking stream.
    const auto state=startConditionalExterior(config.field,entry,direction,rng);
    return traceTrackedFlights(state,Spectrum::Ones(),0,config,nullptr,medium,rng,statistics,renewalMedium);
}

void accumulateGpis(const GpisFlightSample& sample, RenderStatistics& statistics, bool continuation) {
    ++statistics.gpisFlights;
    if (continuation) ++statistics.gpisContinuationFlights;
    statistics.gpisBatches+=sample.batches; statistics.gpisPoints+=sample.points;
    if (sample.status==GpisFlightStatus::RejectedUpcrossing) ++statistics.gpisRejectedUpcrossings;
    if (sample.status==GpisFlightStatus::InvalidGradient) {
        ++statistics.gpisInvalidGradients; ++statistics.numericalFailures;
    }
}

Spectrum traceGpisFlights(Ray ray, TungstenSurfaceState birth, Spectrum throughput, int startDepth,
    const ExperimentConfig& config, const FunctionSpaceGpisMedium& medium,
    TungstenRandom& rng, RenderStatistics& statistics) {
    for (int depth=startDepth;depth<config.render.safetyDepthCap;++depth) {
        const auto hit=medium.sample(ray,birth,rng);
        accumulateGpis(hit,statistics,true);
        if (hit.status!=GpisFlightStatus::Hit) {
            statistics.accumulatedPathDepth+=depth;
            if (hit.status!=GpisFlightStatus::Escape) return Spectrum::Zero();
            ++statistics.escapedPaths;
            return throughput.cwiseProduct(environmentEmission(ray.direction,config.render.environment));
        }
        ++statistics.realCollisions;
        const Vector3 normal=normalizedOrThrow(hit.surface.gradient);
        throughput=throughput.cwiseProduct(conductorFresnel(-ray.direction.dot(normal),config.material.conductor));
        ray={hit.position,normalizedOrThrow(reflectTravelDirection(ray.direction,normal))};
        birth=hit.surface;
        if (depth+1>=config.render.rouletteStartDepth) {
            const double p=std::clamp(throughput.maxCoeff(),0.05,0.95);
            if (rng.uniform01()>=p) {
                ++statistics.rouletteTerminations; statistics.accumulatedPathDepth+=depth+1;
                return Spectrum::Zero();
            }
            throughput/=p;
        }
    }
    ++statistics.safetyCapTerminations;
    statistics.accumulatedPathDepth+=config.render.safetyDepthCap;
    return Spectrum::Zero();
}

class HybridFirstHitTransport {
    const ExperimentConfig& config_;
    GPSSField negativeField_;
    MaterialConfig localFirstMaterial_;
    std::unique_ptr<FirstHitSEMedium> first_;
    std::unique_ptr<FunctionSpaceGpisMedium> tungstenFirst_, tungstenTail_;
    std::unique_ptr<RenewalMedium> positive_, negative_;
    std::unique_ptr<NarrowBandMedium> local_;
    std::unique_ptr<NarrowBandMedium> localFirst_;
    std::unique_ptr<ConditionalMedium> conditionalFirst_;
    std::unique_ptr<GlobalConditionalRenewalPlusMedium> conditionalRenewalFirst_;
    std::unique_ptr<ConditionalMedium> conditionalPositive_, conditionalNegative_;
    std::unique_ptr<GlobalConditionalRenewalPlusMedium> conditionalRenewalPositive_, conditionalRenewalNegative_;

    FirstHitSESurfaceSample sampleLocalFirst(const Ray& ray, Random& eventRng,
        Random& gradientRng, RenderStatistics& statistics) const {
        FirstHitSESurfaceSample hit;
        const auto interval = config_.field.activeDomain.intersect(ray);
        if (!interval.hit || !(interval.exit > interval.entry)) return hit;
        hit.enteredDomain = true;
        hit.entryDistance = interval.entry;
        hit.entry = (ray.origin + interval.entry*ray.direction)
            .cwiseMax(config_.field.activeDomain.minimum).cwiseMin(config_.field.activeDomain.maximum);
        const auto kernel = localFirst_->beginFlight(localFirst_->startExternal(hit.entry,ray.direction));
        hit.maximumDistance = kernel.maximumAgeInDomain();
        ++statistics.classicFlights;
        ++statistics.classicFirstHitFlights;
        const auto sample = localFirst_->sample(kernel,eventRng,config_.numeric,&statistics.tracking);
        hit.hit = sample.collided;
        hit.distance = sample.age;
        const double ell = 1/std::sqrt(ray.direction.dot(config_.field.kernel.precision()*ray.direction));
        hit.q = hit.distance/ell;
        hit.position = hit.entry + hit.distance*ray.direction;
        hit.observationPosition = hit.position;
        if (hit.hit) {
            // Exact local target VNDF, retaining the full gradient required by
            // the GPIS surface birth. The common path applies Fresnel once.
            hit.gradient = sampleCollisionGradient(kernel,hit.distance,gradientRng,config_.numeric);
            hit.normal = normalizedOrThrow(hit.gradient);
            hit.speed = -hit.gradient.dot(ray.direction)*ell/config_.field.kernel.sigma();
        }
        // This is a local extinction event, with no sampled exterior GP state.
        // The hybrid surface handoff assigns F=0 and the original field side.
        return hit;
    }

    FirstHitSESurfaceSample sampleConditionalFirst(const Ray& ray, Random& eventRng,
        Random& gradientRng, RenderStatistics& statistics) const {
        FirstHitSESurfaceSample hit;
        const auto interval = config_.field.activeDomain.intersect(ray);
        if (!interval.hit || !(interval.exit > interval.entry)) return hit;
        hit.enteredDomain = true;
        hit.entryDistance = interval.entry;
        hit.entry = (ray.origin + interval.entry*ray.direction)
            .cwiseMax(config_.field.activeDomain.minimum).cwiseMin(config_.field.activeDomain.maximum);
        // Match the ordinary global_conditional camera birth, including the
        // sampled positive total field and full (unnormalized) prior gradient.
        const auto birth = conditionalFirst_->startExternal(hit.entry,ray.direction,eventRng);
        const auto kernel = conditionalFirst_->beginFlight(birth);
        hit.maximumDistance = kernel.maximumAgeInDomain();
        ++statistics.conditionalFlights;
        ++statistics.conditionalFirstHitFlights;
        if (conditionalRenewalFirst_) {
            std::uint64_t segments = 0;
            const auto sample = conditionalRenewalFirst_->sample(birth,eventRng,&statistics.tracking,
                hit.maximumDistance,&segments,config_.numeric,&gradientRng);
            hit.hit = sample.collided;
            hit.distance = sample.age;
            hit.gradient = sample.gradient;
            statistics.conditionalFirstHitRenewalSegments += segments;
            statistics.conditionalRenewalSegments += segments;
        } else {
            const auto sample = conditionalFirst_->sample(kernel,eventRng,&statistics.tracking);
            hit.hit = sample.collided;
            hit.distance = sample.age;
            if (hit.hit)
                hit.gradient = sampleCollisionGradient(kernel,hit.distance,gradientRng,config_.numeric);
        }
        const double ell = 1/std::sqrt(ray.direction.dot(config_.field.kernel.precision()*ray.direction));
        hit.q = hit.distance/ell;
        hit.position = hit.entry + hit.distance*ray.direction;
        hit.observationPosition = hit.position;
        if (hit.hit) {
            hit.normal = normalizedOrThrow(hit.gradient);
            hit.speed = -hit.gradient.dot(ray.direction)*ell/config_.field.kernel.sigma();
        }
        // initialSign=+1 and fieldValue=0 describe the accepted downcrossing.
        // No Phi(mean/sigma) weight: the camera birth is conditioned on F>0.
        return hit;
    }
public:
    explicit HybridFirstHitTransport(const ExperimentConfig& config)
        : config_(config), negativeField_(firstHitSideField(config.field,-1)),
          localFirstMaterial_(config.material) {
        if (config.firstHit.method==kTungstenFunctionSpaceMethod)
            tungstenFirst_=std::make_unique<FunctionSpaceGpisMedium>(config.field,config.firstHit.tungsten);
        else if (config.firstHit.method=="global_conditional" ||
                 config.firstHit.method=="global_conditional_renewal_plus") {
            conditionalFirst_=std::make_unique<ConditionalMedium>(config.field);
            if (config.firstHit.method=="global_conditional_renewal_plus")
                conditionalRenewalFirst_=std::make_unique<GlobalConditionalRenewalPlusMedium>(
                    config.field,config.firstHit.renewalDistance);
        }
        else if (config.firstHit.method=="classic_local") {
            localFirstMaterial_.gpModel = GpModel::LocalTangent;
            // A global projected-area bound can underestimate the local NDF,
            // whose mean gradient has unit length. Recompute when models differ.
            localFirst_ = std::make_unique<NarrowBandMedium>(config.field,localFirstMaterial_,
                config.mediumDensity,config.densityMajorantGrid,
                config.material.gpModel==GpModel::LocalTangent ? config.preparedAreaMajorant : std::nullopt);
        }
        else first_=std::make_unique<FirstHitSEMedium>(config.field,config.firstHit.model,
            config.firstHit.profileMaximumStep,config.firstHit.reference);
        if (config.continuationMethod == "classic_local") {
            // This is the scene's existing local transport model. Its baked
            // density and tangent NDF keep the original field orientation;
            // it does not carry the first segment's GP conditioning forward.
            local_ = std::make_unique<NarrowBandMedium>(config.field,config.material,
                config.mediumDensity,config.densityMajorantGrid,config.preparedAreaMajorant);
        } else if (config.continuationMethod == "global_conditional") {
            conditionalPositive_ = std::make_unique<ConditionalMedium>(config.field);
            conditionalNegative_ = std::make_unique<ConditionalMedium>(negativeField_);
        } else if (config.continuationMethod == "global_conditional_renewal_plus") {
            conditionalRenewalPositive_ = std::make_unique<GlobalConditionalRenewalPlusMedium>(
                config.field,config.conditionalRenewalDistance);
            conditionalRenewalNegative_ = std::make_unique<GlobalConditionalRenewalPlusMedium>(
                negativeField_,config.conditionalRenewalDistance);
        } else if (config.continuationMethod==kTungstenFunctionSpaceMethod) {
            tungstenTail_=std::make_unique<FunctionSpaceGpisMedium>(config.field,config.continuationTungsten);
        } else {
            positive_ = std::make_unique<RenewalMedium>(config.field,*config.renewal.model,
                config.renewal.profileMaximumStep,config.renewal.profileMode);
            negative_ = std::make_unique<RenewalMedium>(negativeField_,*config.renewal.model,
                config.renewal.profileMaximumStep,config.renewal.profileMode);
        }
    }

    Spectrum trace(const Ray& input, Random& eventRng, Random& gradientRng, Random& continuationRng,
        RenderStatistics& statistics, FirstHitRenderRecord* record = nullptr,
        const FirstHitSESurfaceSample* preparedFirst = nullptr) const {
        const Ray ray{input.origin,normalizedOrThrow(input.direction)};
        std::optional<TungstenRandom> gpisRng;
        if (tungstenFirst_ || tungstenTail_) {
            const auto seed=tungstenFirst_ ? eventRng.integer() : continuationRng.integer();
            gpisRng.emplace(seed);
            if (record) record->gpisSeed=seed;
        }
        FirstHitSESurfaceSample hit;
        if (tungstenFirst_) {
            const auto sample=tungstenFirst_->sample(ray,std::nullopt,*gpisRng);
            accumulateGpis(sample,statistics,false);
            hit.enteredDomain=sample.enteredDomain;
            hit.hit=sample.status==GpisFlightStatus::Hit;
            hit.failed=sample.status!=GpisFlightStatus::Hit && sample.status!=GpisFlightStatus::Escape;
            hit.initialSign=sample.initialSign;
            hit.entryDistance=sample.entryDistance; hit.maximumDistance=sample.maximumDistance;
            hit.distance=sample.distance; hit.entry=sample.entry; hit.position=sample.position;
            hit.gradient=sample.surface.gradient;
            hit.observationPosition=sample.surface.position; hit.fieldValue=sample.surface.value;
            const double ell=1/std::sqrt(ray.direction.dot(config_.field.kernel.metric()*ray.direction));
            hit.q=hit.distance/ell;
            hit.speed=std::abs(hit.gradient.dot(ray.direction))*ell/config_.field.kernel.sigma();
            if (hit.hit) hit.normal=normalizedOrThrow(hit.gradient);
            if (sample.status==GpisFlightStatus::RejectedUpcrossing) ++statistics.firstHitUpcrossings;
        } else {
            if (localFirst_) hit=sampleLocalFirst(ray,eventRng,gradientRng,statistics);
            else if (conditionalFirst_) hit=sampleConditionalFirst(ray,eventRng,gradientRng,statistics);
            else if (preparedFirst) hit=*preparedFirst;
            else {
                const auto flight=first_->beginFlight(ray);
                if (flight) hit=first_->sample(*flight,eventRng,gradientRng,config_.numeric.distanceAbsoluteTolerance);
            }
            hit.observationPosition=hit.position;
            if (hit.hit && hit.initialSign<0) ++statistics.firstHitUpcrossings;
            if (tungstenTail_ && hit.hit && hit.gradient.dot(ray.direction)>0) {
                hit.hit=false; hit.failed=true;
                ++statistics.gpisRejectedUpcrossings;
            } else if (tungstenTail_ && hit.hit && (!hit.gradient.allFinite() ||
                       hit.gradient.squaredNorm()<static_cast<double>(0.0000001f))) {
                hit.hit=false; hit.failed=true;
                ++statistics.gpisInvalidGradients; ++statistics.numericalFailures;
            }
        }
        if (hit.enteredDomain) ++statistics.firstHitFlights;
        else ++statistics.firstHitDomainMisses;
        if (record) { record->cameraRay = ray; record->first = hit; }
        if (hit.failed) return Spectrum::Zero();
        if (!hit.hit) {
            if (hit.enteredDomain) ++statistics.firstHitEscapes;
            ++statistics.escapedPaths;
            return environmentEmission(ray.direction,config_.render.environment);
        }
        ++statistics.firstHits; ++statistics.realCollisions;
        RenewalPathState state;
        state.throughput = conductorFresnel(-ray.direction.dot(hit.normal),config_.material.conductor);
        state.ray = {hit.position,normalizedOrThrow(reflectTravelDirection(ray.direction,hit.normal))};
        // An accepted Tungsten hit has the original field orientation, even
        // when earlier no-hit batches started on the negative side.
        const int side=(tungstenFirst_ || tungstenTail_) ? 1 : hit.initialSign;
        state.gradient = side*hit.gradient;
        // The first event already sampled survival, distance and the collision
        // gradient. Only Fresnel is applied here, with no second phase sample.
        if (state.depth >= config_.render.rouletteStartDepth) {
            const double p = std::clamp(state.throughput.maxCoeff(),0.05,0.95);
            if ((gpisRng ? gpisRng->uniform01() : continuationRng.openUniform01()) >= p) {
                ++statistics.rouletteTerminations; statistics.accumulatedPathDepth += state.depth;
                return Spectrum::Zero();
            }
            state.throughput /= p;
        }
        if (tungstenTail_) return traceGpisFlights(state.ray,
            {hit.observationPosition,hit.fieldValue,hit.gradient},state.throughput,state.depth,
            config_,*tungstenTail_,*gpisRng,statistics);
        if (local_) return traceTrackedFlights(startClassicCollisionFlight(state.ray.origin,state.ray.direction),
            state.throughput,state.depth,config_,local_.get(),nullptr,continuationRng,statistics);
        if (conditionalPositive_ || conditionalRenewalPositive_) {
            FlightState birth = startClassicCollisionFlight(state.ray.origin,state.ray.direction);
            birth.birthValue = 0.0;
            birth.birthGradient = state.gradient;
            // Keep the full first-hit gradient and its side. This is a surface
            // birth, not a new exterior observation or a second phase sample.
            const auto* medium = side > 0 ? conditionalPositive_.get() : conditionalNegative_.get();
            const auto* renewalMedium = side > 0 ? conditionalRenewalPositive_.get() : conditionalRenewalNegative_.get();
            return traceTrackedFlights(birth,state.throughput,state.depth,config_,nullptr,medium,
                continuationRng,statistics,renewalMedium);
        }
        return traceRenewalFromSurface(state,config_,side > 0 ? *positive_ : *negative_,
            continuationRng,statistics);
    }
};

// First-zero GPU inference has its own state contract. Reuse the CPU executor
// and the hybrid continuation, but keep its device states separate from the
// surface-born renewal GRU. Batches bound memory independently of image size.
RenderedImage renderCausalFirstHit(const ExperimentConfig& config,const HybridFirstHitTransport& hybrid,
    std::atomic<std::uint64_t>* completed) {
    RenderedImage result; result.width=config.render.width; result.height=config.render.height;
    result.pixels.assign(static_cast<std::size_t>(result.width)*result.height,Spectrum::Zero());
    const std::size_t spp=config.render.samplesPerPixel,totalSamples=result.pixels.size()*spp;
    if (config.render.writeFirstHitSamples) {
        if (totalSamples>1000000) throw std::invalid_argument("first-hit CSV diagnostics are limited to 1,000,000 camera samples");
        result.firstHitRecords.resize(totalSamples);
    }
    const Vector3 forward=normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
    Vector3 right,up; orthonormalComplement(forward,right,up);
    const double aspect=static_cast<double>(result.width)/result.height,scale=std::tan(config.render.verticalFovDegrees*kPi/360.);
    FirstHitSEMedium medium(config.field,config.firstHit.model,config.firstHit.profileMaximumStep,config.firstHit.reference);
    CausalFirstHitSession session(*config.firstHit.model->causal(),config.firstHit.backend,config.render.threadCount);
    RenewalCpuExecutor executor(config.render.threadCount,config.firstHit.batchSize);
    for (std::size_t begin=0;begin<totalSamples;begin+=config.firstHit.batchSize) {
        const std::size_t count=std::min<std::size_t>(config.firstHit.batchSize,totalSamples-begin);
        std::vector<Ray> rays(count);
        std::vector<std::optional<FirstHitSEFlight>> flights(count);
        std::vector<CausalFirstHitUniforms> allUniforms(count);
        executor.forRanges(count,[&](std::size_t a,std::size_t b) {
            for (std::size_t i=a;i<b;++i) {
                const auto pixel=(begin+i)/spp,sample=(begin+i)%spp;
                const auto x=pixel%result.width,y=pixel/result.width;
                Random camera(firstHitStreamSeed(config.seed,pixel,sample,0)),event(firstHitStreamSeed(config.seed,pixel,sample,1));
                const double px=(2*((x+camera.openUniform01())/result.width)-1)*aspect*scale;
                const double py=(1-2*((y+camera.openUniform01())/result.height))*scale;
                rays[i]={config.render.cameraPosition,normalizedOrThrow(forward+px*right+py*up)};
                flights[i]=medium.beginFlight(rays[i]); allUniforms[i]=causalFirstHitUniforms(event);
            }
        });
        std::vector<const RayMeanProfile*> profiles; std::vector<CausalFirstHitUniforms> uniforms;
        std::vector<std::size_t> slots;
        double tolerance=std::numeric_limits<double>::infinity();
        for (std::size_t i=0;i<count;++i) if (flights[i]) {
            profiles.push_back(&flights[i]->mean); uniforms.push_back(allUniforms[i]); slots.push_back(i);
            tolerance=std::min(tolerance,config.numeric.distanceAbsoluteTolerance/flights[i]->ell);
        }
        std::vector<FirstHitSESample> draws(count);
        if (!profiles.empty()) {
            const auto sampled=session.sample(profiles,uniforms,tolerance);
            for (std::size_t i=0;i<slots.size();++i) draws[slots[i]]=sampled[i];
        }
        std::vector<Spectrum> radiance(count);
        std::vector<RenderStatistics> statistics(count);
        executor.forRanges(count,[&](std::size_t a,std::size_t b) {
            for (std::size_t i=a;i<b;++i) {
                const auto pixel=(begin+i)/spp,sample=(begin+i)%spp;
                Random event(firstHitStreamSeed(config.seed,pixel,sample,1)),gradient(firstHitStreamSeed(config.seed,pixel,sample,2)),
                    continuation(firstHitStreamSeed(config.seed,pixel,sample,3));
                auto* record=result.firstHitRecords.empty() ? nullptr : &result.firstHitRecords[begin+i];
                FirstHitSESurfaceSample first;
                try {
                    if (flights[i]) first=medium.reconstruct(*flights[i],draws[i],gradient);
                    ++statistics[i].paths;
                    radiance[i]=hybrid.trace(rays[i],event,gradient,continuation,statistics[i],record,&first);
                } catch (const std::exception& error) {
                    throw std::runtime_error("causal first-hit pixel="+std::to_string(pixel)+", sample="+
                        std::to_string(sample)+": "+error.what());
                }
                if (record) record->radiance=radiance[i];
            }
        });
        for (std::size_t i=0;i<count;++i) {
            result.pixels[(begin+i)/spp]+=radiance[i]; mergeInto(result.statistics,statistics[i]);
        }
        if (completed) completed->fetch_add(count,std::memory_order_relaxed);
    }
    for (auto& pixel:result.pixels) pixel/=static_cast<double>(spp);
    return result;
}

} // namespace

Spectrum traceRenewalFromSurface(const RenewalPathState& state, const ExperimentConfig& config,
    const RenewalMedium& medium, Random& rng, RenderStatistics& statistics) {
    if (state.depth < 1 || !state.gradient.allFinite() || !state.throughput.allFinite() ||
        !(state.ray.direction.dot(state.gradient) > 0))
        throw std::invalid_argument("invalid outward renewal continuation state");
    Ray ray{state.ray.origin,normalizedOrThrow(state.ray.direction)};
    return traceRenewalFlights(ray,state.gradient,state.throughput,state.depth,config,medium,rng,statistics);
}

Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, Random& rng,
                         RenderStatistics& statistics) {
    if (config.transportMode == "global_conditional_renewal_plus") {
        ExperimentConfig prepared = config;
        prepareRenewalModel(prepared);
        const GlobalConditionalRenewalPlusMedium medium(prepared.field,prepared.conditionalRenewalDistance);
        return traceConditionalCameraPath(initialRay,prepared,nullptr,rng,statistics,&medium);
    }
    if (config.transportMode == "hybrid_first_hit") {
        ExperimentConfig prepared = config;
        prepareRenewalModel(prepared); prepareFirstHitModel(prepared);
        const HybridFirstHitTransport hybrid(prepared);
        const auto seed = rng.integer();
        Random eventRng(firstHitStreamSeed(seed,0,0,1)), gradientRng(firstHitStreamSeed(seed,0,0,2)),
               continuationRng(firstHitStreamSeed(seed,0,0,3));
        return hybrid.trace(initialRay,eventRng,gradientRng,continuationRng,statistics);
    }
    if (config.transportMode == "neural_renewal") {
        ExperimentConfig prepared = config;
        prepareRenewalModel(prepared);
        const RenewalMedium medium(prepared.field, *prepared.renewal.model, prepared.renewal.profileMaximumStep,
                                   prepared.renewal.profileMode);
        return traceRenewalCameraPath(initialRay, prepared, medium, rng, statistics);
    }
    const NarrowBandMedium medium(config.field, config.material, config.mediumDensity,
                                  config.densityMajorantGrid, config.preparedAreaMajorant);
    return traceCameraPath(initialRay, config, medium, rng, statistics);
}

Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, const NarrowBandMedium& medium,
                         Random& rng, RenderStatistics& statistics) {
    if (config.transportMode == "neural_renewal" || config.transportMode == "hybrid_first_hit" ||
        config.transportMode == "global_conditional_renewal_plus")
        return traceCameraPath(initialRay, config, rng, statistics);
    if (config.transportMode == "global_conditional") {
        const ConditionalMedium conditional(config.field);
        return traceConditionalCameraPath(initialRay,config,&conditional,rng,statistics);
    }
    const Vector3 initialDirection = normalizedOrThrow(initialRay.direction);
    const DomainInterval firstInterval = config.field.activeDomain.intersect(
        {initialRay.origin, initialDirection});
    if (!firstInterval.hit) {
        ++statistics.escapedPaths;
        return environmentEmission(initialDirection, config.render.environment);
    }
    const Point3 entry = initialRay.origin + firstInterval.entry * initialDirection;
    const FlightState state = medium.startExternal(entry, initialDirection);
    return traceTrackedFlights(state,Spectrum::Ones(),0,config,&medium,nullptr,rng,statistics);
}

RenderedImage renderAnalyticScene(const ExperimentConfig& config,
                                 std::atomic<std::uint64_t>* completedCameraRays) {
    // Prepare the local extinction bound before tracing any paths. All worker
    // threads share this immutable medium and its baked DDA majorant.
    ExperimentConfig prepared = config;
    std::unique_ptr<NarrowBandMedium> medium;
    std::unique_ptr<RenewalMedium> renewal;
    std::unique_ptr<HybridFirstHitTransport> hybrid;
    std::unique_ptr<GlobalConditionalRenewalPlusMedium> conditionalRenewal;
    if (config.transportMode == "hybrid_first_hit") {
        prepareRenewalModel(prepared); prepareFirstHitModel(prepared);
        hybrid = std::make_unique<HybridFirstHitTransport>(prepared);
        if (prepared.firstHit.backend!="scalar")
            return renderCausalFirstHit(prepared,*hybrid,completedCameraRays);
    } else if (config.transportMode == "neural_renewal") {
        prepareRenewalModel(prepared);
        if (prepared.renewal.resolvedBackend != "scalar")
            return renderRenewalWavefront(prepared,completedCameraRays);
        renewal = std::make_unique<RenewalMedium>(prepared.field, *prepared.renewal.model,
                                                prepared.renewal.profileMaximumStep, prepared.renewal.profileMode);
    } else if (config.transportMode == "global_conditional_renewal_plus") {
        prepareRenewalModel(prepared);
        conditionalRenewal = std::make_unique<GlobalConditionalRenewalPlusMedium>(
            prepared.field,prepared.conditionalRenewalDistance);
    } else {
        medium = std::make_unique<NarrowBandMedium>(config.field, config.material, config.mediumDensity,
                                                   config.densityMajorantGrid, config.preparedAreaMajorant);
    }
    RenderedImage result;
    result.width = config.render.width;
    result.height = config.render.height;
    result.pixels.assign(static_cast<std::size_t>(result.width * result.height), Spectrum::Zero());
    if (hybrid && config.render.writeFirstHitSamples) {
        const auto count = result.pixels.size()*static_cast<std::size_t>(config.render.samplesPerPixel);
        if (count > 1000000) throw std::invalid_argument("first-hit CSV diagnostics are limited to 1,000,000 camera samples");
        result.firstHitRecords.resize(count);
    }
    const Vector3 forward = normalizedOrThrow(config.render.cameraTarget - config.render.cameraPosition);
    Vector3 right, up;
    orthonormalComplement(forward, right, up);
    const double aspect = static_cast<double>(result.width) / result.height;
    const double scale = std::tan(config.render.verticalFovDegrees * kPi / 360.0);
    // Every pixel seeds its own stream from its index alone, so distributing rows over
    // workers leaves the image bit-identical to the serial result.
    const int workers = resolveWorkerCount(config.render.threadCount, result.height);
    std::vector<RenderStatistics> perWorker(static_cast<std::size_t>(workers));
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(workers));
    std::atomic<int> nextRow{0};
    std::atomic<bool> aborted{false};

    const auto renderRow = [&](int worker, int y) {
        RenderStatistics& statistics = perWorker[static_cast<std::size_t>(worker)];
        for (int x = 0; x < result.width; ++x) {
            Spectrum sum = Spectrum::Zero();
            const std::uint64_t pixelIndex = static_cast<std::uint64_t>(y * result.width + x);
            Random rng(config.seed + 0x9e3779b97f4a7c15ULL * (pixelIndex + 1));
            for (int sample = 0; sample < config.render.samplesPerPixel; ++sample) {
                std::optional<Random> cameraRng;
                if (hybrid) cameraRng.emplace(firstHitStreamSeed(config.seed,pixelIndex,sample,0));
                Random& jitterRng = hybrid ? *cameraRng : rng;
                const double px = (2.0 * ((x + jitterRng.openUniform01()) / result.width) - 1.0) *
                                  aspect * scale;
                const double py = (1.0 - 2.0 * ((y + jitterRng.openUniform01()) / result.height)) * scale;
                const Vector3 direction = normalizedOrThrow(forward + px * right + py * up);
                ++statistics.paths;
                const Ray ray{config.render.cameraPosition, direction};
                if (hybrid) {
                    Random eventRng(firstHitStreamSeed(config.seed,pixelIndex,sample,1));
                    Random gradientRng(firstHitStreamSeed(config.seed,pixelIndex,sample,2));
                    Random continuationRng(firstHitStreamSeed(config.seed,pixelIndex,sample,3));
                    auto* record = result.firstHitRecords.empty() ? nullptr :
                        &result.firstHitRecords[pixelIndex*config.render.samplesPerPixel+sample];
                    Spectrum radiance;
                    try {
                        radiance = hybrid->trace(ray,eventRng,gradientRng,continuationRng,statistics,record);
                    } catch (const std::exception& error) {
                        throw std::runtime_error("hybrid_first_hit pixel=("+std::to_string(x)+","+
                            std::to_string(y)+"), sample="+std::to_string(sample)+", seed="+
                            std::to_string(config.seed)+": "+error.what());
                    }
                    if (record) record->radiance = radiance;
                    sum += radiance;
                } else if (conditionalRenewal)
                    sum += traceConditionalCameraPath(ray,prepared,nullptr,rng,statistics,conditionalRenewal.get());
                else sum += renewal ? traceRenewalCameraPath(ray, prepared, *renewal, rng, statistics)
                                      : traceCameraPath(ray, config, *medium, rng, statistics);
                if (completedCameraRays)
                    completedCameraRays->fetch_add(1, std::memory_order_relaxed);
            }
            result.pixels[static_cast<std::size_t>(y * result.width + x)] =
                sum / config.render.samplesPerPixel;
        }
    };

    // Dynamic row assignment: cost per row varies by orders of magnitude across a
    // image, so static chunks would leave workers idle.
    const auto drainRows = [&](int worker) {
        for (;;) {
            const int y = nextRow.fetch_add(1, std::memory_order_relaxed);
            if (y >= result.height || aborted.load(std::memory_order_relaxed)) return;
            renderRow(worker, y);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers) - 1);
    try {
        for (int worker = 1; worker < workers; ++worker) {
            pool.emplace_back([&, worker] {
                try {
                    drainRows(worker);
                } catch (...) {
                    failures[static_cast<std::size_t>(worker)] = std::current_exception();
                    aborted.store(true, std::memory_order_relaxed);
                }
            });
        }
    } catch (...) {
        aborted.store(true, std::memory_order_relaxed);
        for (std::thread& thread : pool) {
            if (thread.joinable()) thread.join();
        }
        throw;
    }
    try {
        drainRows(0);
    } catch (...) {
        failures[0] = std::current_exception();
        aborted.store(true, std::memory_order_relaxed);
    }
    for (std::thread& thread : pool) thread.join();
    for (const std::exception_ptr& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }
    for (const RenderStatistics& statistics : perWorker) mergeInto(result.statistics, statistics);
    return result;
}

void writePfm(const std::filesystem::path& path, const RenderedImage& image) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open PFM output");
    stream << "PF\n" << image.width << ' ' << image.height << "\n-1.0\n";
    for (int y = image.height - 1; y >= 0; --y) {
        for (int x = 0; x < image.width; ++x) {
            const Spectrum& pixel = image.pixels[static_cast<std::size_t>(y * image.width + x)];
            const float rgb[3]{static_cast<float>(pixel.x()), static_cast<float>(pixel.y()),
                               static_cast<float>(pixel.z())};
            stream.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
    }
}

void writeBmpPreview(const std::filesystem::path& path, const RenderedImage& image) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open BMP preview output");
    auto write16 = [&stream](std::uint16_t value) {
        const unsigned char bytes[2]{static_cast<unsigned char>(value),
                                     static_cast<unsigned char>(value >> 8)};
        stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    };
    auto write32 = [&stream](std::uint32_t value) {
        const unsigned char bytes[4]{static_cast<unsigned char>(value),
                                     static_cast<unsigned char>(value >> 8),
                                     static_cast<unsigned char>(value >> 16),
                                     static_cast<unsigned char>(value >> 24)};
        stream.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    };
    const std::uint32_t rowBytes = static_cast<std::uint32_t>((image.width * 3 + 3) & ~3);
    const std::uint32_t pixelBytes = rowBytes * static_cast<std::uint32_t>(image.height);
    stream.write("BM", 2);
    write32(54 + pixelBytes);
    write32(0);
    write32(54);
    write32(40);
    write32(static_cast<std::uint32_t>(image.width));
    write32(static_cast<std::uint32_t>(image.height));
    write16(1);
    write16(24);
    write32(0);
    write32(pixelBytes);
    write32(2835);
    write32(2835);
    write32(0);
    write32(0);
    auto encode = [](double linear) {
        const double x = std::clamp(linear, 0.0, 1.0);
        const double srgb = x <= 0.0031308 ? 12.92 * x
                                           : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
        return static_cast<unsigned char>(std::lround(255.0 * srgb));
    };
    const unsigned char padding[3]{0, 0, 0};
    const std::uint32_t paddingBytes = rowBytes - static_cast<std::uint32_t>(image.width * 3);
    for (int y = image.height - 1; y >= 0; --y) {
        for (int x = 0; x < image.width; ++x) {
            const Spectrum& pixel = image.pixels[static_cast<std::size_t>(y * image.width + x)];
            const unsigned char bgr[3]{encode(pixel.z()), encode(pixel.y()), encode(pixel.x())};
            stream.write(reinterpret_cast<const char*>(bgr), sizeof(bgr));
        }
        stream.write(reinterpret_cast<const char*>(padding), paddingBytes);
    }
}

} // namespace mf

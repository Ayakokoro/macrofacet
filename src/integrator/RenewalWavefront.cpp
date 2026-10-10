#include "macrofacet/integrator/RenewalWavefront.h"
#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include "macrofacet/learned/RenewalBatchSession.h"
#include "macrofacet/macrofacet/ConductorPhase.h"
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>

namespace mf {
namespace {
Spectrum emission(const Vector3& direction, const std::string& environment) {
    if (environment == "unit_white") return Spectrum::Ones();
    const double up = 0.5*(normalizedOrThrow(direction).z()+1.0);
    return Spectrum(0.15+0.85*up,0.2+0.7*up,0.35+0.55*up);
}
enum class SlotStage { Refill, Initialize, Segment, Mixture, SegmentInFlight, MixtureInFlight, Retired };
struct Slot {
    SlotStage stage = SlotStage::Refill;
    double hitCoordinate = 0;
    std::size_t pixel = 0;
    int sample = 0, depth = 0;
    bool occupied = false, newSample = true;
    Random rng;
    Ray ray;
    Spectrum throughput = Spectrum::Ones();
    std::optional<Vector3> gradient;
    std::optional<RenewalFlight> flight;
    std::size_t segment = 0;
    double remaining = 0;
    RenderStatistics statistics;
};
struct BatchWork {
    RenewalBatchRequests requests;
    RenewalBatchResults output;
    RenewalBatchSession::Ticket ticket = 0;
    bool otherGpuPending = false;
    BatchWork* nextFree = nullptr;
    void reset() {
        requests.initializeSlots.clear(); requests.segmentSlots.clear(); requests.mixtureSlots.clear();
        requests.firstSegments.clear(); requests.segments.clear(); requests.starts.clear(); requests.coordinates.clear();
        // Keep output's high-water size so nested mixture buffers survive reuse.
        ticket = 0; otherGpuPending = false;
    }
};
class BatchWorkPool {
    std::mutex mutex_;
    std::vector<std::unique_ptr<BatchWork>> storage_;
    BatchWork* free_ = nullptr;
public:
    std::shared_ptr<BatchWork> acquire() {
        BatchWork* batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (free_) {
                batch = free_; free_ = batch->nextFree;
            } else {
                storage_.push_back(std::make_unique<BatchWork>());
                batch = storage_.back().get();
            }
        }
        batch->reset();
        // Last-owner release includes CPU tasks, not just the GPU ticket. The
        // mutex orders prior accesses before reuse. Returning a node allocates
        // nothing; the pool outlives all queues and captured worker handles.
        return std::shared_ptr<BatchWork>(batch,[this](BatchWork* finished) {
            std::lock_guard<std::mutex> lock(mutex_);
            finished->nextFree = free_; free_ = finished;
        });
    }
};
}

RenderedImage renderRenewalWavefront(const ExperimentConfig& config,
    std::atomic<std::uint64_t>* completedCameraRays) {
    RenderedImage image;
    image.width = config.render.width; image.height = config.render.height;
    const auto pixelCount = static_cast<std::size_t>(image.width)*image.height;
    image.pixels.assign(pixelCount,Spectrum::Zero());
    const int requestedPool = config.renewal.rayPoolSize ? config.renewal.rayPoolSize :
        std::min(65536,2*config.renewal.batchSize);
    const int capacity = static_cast<int>(std::min<std::size_t>(requestedPool,pixelCount));
    const auto batchSize = static_cast<std::size_t>(std::min(capacity,config.renewal.batchSize));
    const auto auxiliaryMinimum = std::min(batchSize,static_cast<std::size_t>(config.renewal.auxiliaryBatchMinimum));
    const int pipelineDepth = config.renewal.resolvedBackend == "torch_cuda" ?
        config.renewal.maximumInFlightBatches : 1;
    // A separate inference thread exclusively owns LibTorch. This coordinator
    // can dispatch completed CPU work even while an ATen submission is running.
    const RenewalMedium medium(config.field,*config.renewal.model,config.renewal.profileMaximumStep,
                               config.renewal.profileMode);
    auto& statistics = image.statistics;
    auto& d = statistics.renewal;
    d.rayPoolSize = capacity;
    const Vector3 forward = normalizedOrThrow(config.render.cameraTarget-config.render.cameraPosition);
    Vector3 right, up;
    orthonormalComplement(forward,right,up);
    const double aspect = static_cast<double>(image.width)/image.height;
    const double scale = std::tan(config.render.verticalFovDegrees*kPi/360.0);
    std::vector<Slot> slots(static_cast<std::size_t>(capacity));
    // Separate arrays keep the hot segment features contiguous without growing
    // Slot. No resizing after workers start; only the owning worker writes a row.
    RenewalBatchSession::WavefrontFeatures preparedFeatures;
    preparedFeatures.initial.resize(capacity);
    preparedFeatures.segment.resize(capacity);
    preparedFeatures.mixture.resize(capacity);
    std::atomic<std::size_t> nextPixel{0};
    std::atomic<bool> cancelled{false};
    struct Pending { int id; std::uint64_t round; };
    std::deque<Pending> initializationQueue, segmentQueue, mixtureQueue;
    std::size_t outstandingInference = 0;
    std::uint64_t round = 0;

    // Publish once per chunk. Release/acquire transfers exclusive slot ownership.
    // No field query or inference runs under this mutex.
    BatchWorkPool batchPool; // Declared before queues and workers: destroyed last.
    std::mutex readyMutex;
    std::condition_variable readyCondition;
    std::condition_variable inferenceCondition;
    std::deque<std::vector<int>> incoming, received;
    std::deque<std::shared_ptr<BatchWork>> inferenceQueue, completedInference, receivedInference;
    std::size_t outstandingCpuTasks = 0;
    bool stoppingInference = false;
    std::uint64_t maximumGpuPending = 0, blockingCollects = 0;
    std::exception_ptr failure;
    std::thread inferenceThread;
    RenewalCpuExecutor cpu(config.render.threadCount,static_cast<std::size_t>(capacity));
    d.cpuWorkers = cpu.workerCount();
    const auto clearFlight = [&](Slot& slot) {
        slot.flight.reset();
    };
    const auto finish = [&](Slot& slot, const Spectrum& radiance) {
        // Keep one pixel in one slot for all spp: RNG and accumulation order
        // depend on that pixel, not on task completion or batch membership.
        image.pixels[slot.pixel] += radiance;
        slot.statistics.accumulatedPathDepth += slot.depth;
        clearFlight(slot); slot.newSample = true; slot.stage = SlotStage::Refill;
        if (++slot.sample == config.render.samplesPerPixel) {
            image.pixels[slot.pixel] /= config.render.samplesPerPixel;
            slot.occupied = false;
        }
        if (completedCameraRays) completedCameraRays->fetch_add(1,std::memory_order_relaxed);
    };
    const auto escape = [&](Slot& slot) {
        ++slot.statistics.escapedPaths;
        finish(slot,slot.throughput.cwiseProduct(emission(slot.ray.direction,config.render.environment)));
    };
    const auto initializeSlot = [&](Slot& slot) {
        if (slot.stage != SlotStage::Refill) throw std::logic_error("Renewal slot released twice");
        while (!slot.flight && !cancelled.load(std::memory_order_relaxed)) {
            if (!slot.occupied) {
                const auto pixel = nextPixel.fetch_add(1,std::memory_order_relaxed);
                if (pixel >= pixelCount) { slot.stage = SlotStage::Retired; return; }
                slot.pixel = pixel; slot.sample = 0;
                slot.rng = Random(config.seed+0x9e3779b97f4a7c15ULL*(slot.pixel+1));
                slot.occupied = true; slot.newSample = true;
            }
            if (slot.newSample) {
                slot.newSample = false; slot.depth = 0;
                slot.gradient.reset(); slot.throughput = Spectrum::Ones();
                ++slot.statistics.paths;
                const int px = static_cast<int>(slot.pixel%image.width);
                const int py = static_cast<int>(slot.pixel/image.width);
                const double u = (2.0*((px+slot.rng.openUniform01())/image.width)-1.0)*aspect*scale;
                const double v = (1.0-2.0*((py+slot.rng.openUniform01())/image.height))*scale;
                const Vector3 direction = normalizedOrThrow(forward+u*right+v*up);
                slot.ray = {config.render.cameraPosition,normalizedOrThrow(direction)};
                const auto domain = config.field.activeDomain.intersect(slot.ray);
                if (!domain.hit || !(domain.exit > domain.entry)) { escape(slot); continue; }
                slot.ray.origin += domain.entry*slot.ray.direction;
                slot.ray.origin = slot.ray.origin.cwiseMax(config.field.activeDomain.minimum)
                    .cwiseMin(config.field.activeDomain.maximum);
            }
            slot.flight = medium.beginFlight(slot.ray,slot.gradient);
            if (!slot.flight) { escape(slot); continue; }
            slot.segment = 0; slot.remaining = -std::log(slot.rng.openUniform01());
            ++slot.statistics.renewal.flights;
            slot.stage = SlotStage::Initialize;
        }
    };
    const auto advanceSlot = [&](Slot& slot, const RayMeanSegment& segment, const std::array<double,4>& rates) {
        if (slot.stage != SlotStage::SegmentInFlight)
            throw std::logic_error("Renewal segment result does not own slot");
        const RenewalHazardSegment hazard(segment.begin,segment.end,rates);
        const double mass = hazard.integral(0,1);
        if (mass > 0 && slot.remaining <= mass) {
            slot.hitCoordinate = hazard.inverse(slot.remaining,0,1,
                config.numeric.distanceAbsoluteTolerance/slot.flight->ell);
            slot.stage = SlotStage::Mixture;
        } else {
            slot.remaining -= mass;
            if (slot.flight->mean.hasSegment(++slot.segment)) slot.stage = SlotStage::Segment;
            else { escape(slot); initializeSlot(slot); }
        }
    };
    const auto scatterSlot = [&](Slot& slot, double coordinate, const RenewalSpeedMixture& mixture) {
        if (slot.stage != SlotStage::MixtureInFlight)
            throw std::logic_error("Renewal mixture result does not own slot");
        const auto hit = medium.sampleHit(*slot.flight,slot.segment,coordinate,mixture,slot.rng);
        ++slot.statistics.realCollisions; ++slot.depth;
        slot.throughput = slot.throughput.cwiseProduct(conductorFresnel(
            -slot.ray.direction.dot(hit.normal),config.material.conductor));
        const Vector3 outgoing = normalizedOrThrow(reflectTravelDirection(slot.ray.direction,hit.normal));
        if (!(outgoing.dot(hit.gradient) > 0))
            throw std::runtime_error("Renewal reflection must leave the sampled surface");
        slot.ray = {hit.position,outgoing}; slot.gradient = hit.gradient;
        clearFlight(slot); slot.stage = SlotStage::Refill;
        if (slot.depth >= config.render.rouletteStartDepth) {
            const double probability = std::clamp(slot.throughput.maxCoeff(),0.05,0.95);
            if (slot.rng.openUniform01() >= probability) {
                ++slot.statistics.rouletteTerminations; finish(slot,Spectrum::Zero());
                initializeSlot(slot); return;
            }
            slot.throughput /= probability;
        }
        if (slot.depth == config.render.safetyDepthCap) {
            ++slot.statistics.safetyCapTerminations; finish(slot,Spectrum::Zero());
        }
        initializeSlot(slot);
    };
    const auto prepareFeatures = [&](const std::vector<int>& ready) {
        for (const int id : ready) {
            const auto& slot = slots[id];
            switch (slot.stage) {
            case SlotStage::Initialize: {
                const auto& first = slot.flight->mean.segment(0);
                preparedFeatures.initial[id] = RenewalBatchSession::initialFeatures(first,slot.flight->start);
                preparedFeatures.segment[id] = first.features();
                break;
            }
            case SlotStage::Segment:
                preparedFeatures.segment[id] = slot.flight->mean.segment(slot.segment).features();
                break;
            case SlotStage::Mixture:
                preparedFeatures.mixture[id] = RenewalBatchSession::mixtureFeatures(
                    slot.flight->mean.segment(slot.segment),slot.hitCoordinate);
                break;
            default: throw std::logic_error("invalid Renewal feature preparation stage");
            }
        }
    };
    using CpuRange = std::function<void(std::size_t,std::size_t,std::vector<int>&)>;
    const auto enqueueRanges = [&](std::size_t count, CpuRange function) {
        for (std::size_t begin=0; begin<count; begin+=RenewalCpuExecutor::grain) {
            const auto end = std::min(count,begin+RenewalCpuExecutor::grain);
            { std::lock_guard<std::mutex> lock(readyMutex); ++outstandingCpuTasks; }
            try {
                cpu.enqueue([&,begin,end,function] {
                    std::vector<int> ready;
                    std::exception_ptr error;
                    try {
                        if (!cancelled.load(std::memory_order_relaxed)) {
                            ready.reserve(end-begin);
                            function(begin,end,ready);
                            // Compute before publishing this chunk. The ready
                            // mutex transfers visibility to the scheduler and
                            // inference thread; no extra task or barrier is used.
                            prepareFeatures(ready);
                        }
                    } catch (...) { error = std::current_exception(); }
                    {
                        std::lock_guard<std::mutex> lock(readyMutex);
                        try { if (!error && !ready.empty()) incoming.push_back(std::move(ready)); }
                        catch (...) { error = std::current_exception(); }
                        if (error) {
                            if (!failure) failure = error;
                            cancelled.store(true,std::memory_order_relaxed);
                        }
                        --outstandingCpuTasks;
                    }
                    readyCondition.notify_one();
                });
            } catch (...) {
                std::lock_guard<std::mutex> lock(readyMutex); --outstandingCpuTasks; throw;
            }
        }
    };
    const auto drainIncoming = [&]() {
        std::size_t active;
        {
            std::lock_guard<std::mutex> lock(readyMutex);
            if (failure) std::rethrow_exception(failure);
            received.swap(incoming); active = outstandingCpuTasks;
            receivedInference.swap(completedInference);
        }
        for (const auto& chunk : received) for (int id : chunk) {
            switch (slots[id].stage) {
            case SlotStage::Initialize: initializationQueue.push_back({id,round}); break;
            case SlotStage::Segment: segmentQueue.push_back({id,round}); break;
            case SlotStage::Mixture: mixtureQueue.push_back({id,round}); break;
            default: throw std::logic_error("invalid published Renewal slot");
            }
        }
        received.clear();
        return active;
    };
    const auto prepareBatch = [&](RenewalBatchRequests& requests, std::size_t activeCpu) {
        const bool idle = !activeCpu && !outstandingInference;
        const auto aged = [&](const std::deque<Pending>& queue) {
            return !queue.empty() && round-queue.front().round >=
                static_cast<std::uint64_t>(config.renewal.maximumQueueDelay);
        };
        const auto ready = [&](const std::deque<Pending>& queue) {
            return !queue.empty() && (queue.size() >= auxiliaryMinimum || idle || aged(queue));
        };
        // All request groups share one distinct-ray budget. Taking a full
        // segment group plus extra mixture rays can drain a two-batch ray pool,
        // leaving too few independent slots to keep the other group moving.
        const auto nm = ready(mixtureQueue) ? std::min(batchSize,mixtureQueue.size()) : 0;
        const auto ni = ready(initializationQueue) ? std::min(batchSize-nm,initializationQueue.size()) : 0;
        const auto ns = std::min(batchSize-nm-ni,segmentQueue.size());
        const auto total = ni+ns+nm;
        // Do not turn freshly published CPU chunks into small GPU requests
        // while other CPU tasks or GPU batches can replenish the queues.
        if (!total || (!idle && total < batchSize)) return false;
        requests.initializeSlots.reserve(ni); requests.firstSegments.reserve(ni); requests.starts.reserve(ni);
        requests.segmentSlots.reserve(ni+ns); requests.segments.reserve(ni+ns);
        requests.mixtureSlots.reserve(nm); requests.coordinates.reserve(nm);
        for (std::size_t i=0;i<nm;++i) {
            const auto pending = mixtureQueue.front(); mixtureQueue.pop_front();
            auto& slot = slots[pending.id];
            if (slot.stage != SlotStage::Mixture) throw std::logic_error("invalid Renewal mixture queue");
            requests.mixtureSlots.push_back(pending.id); requests.coordinates.push_back(slot.hitCoordinate);
            slot.stage = SlotStage::MixtureInFlight;
            d.maximumMixtureWait = std::max(d.maximumMixtureWait,round-pending.round);
        }
        for (std::size_t i=0;i<ni;++i) {
            const auto pending = initializationQueue.front(); initializationQueue.pop_front();
            auto& slot = slots[pending.id];
            if (slot.stage != SlotStage::Initialize) throw std::logic_error("invalid Renewal initialization queue");
            requests.initializeSlots.push_back(pending.id);
            requests.firstSegments.push_back(slot.flight->mean.segment(0));
            requests.starts.push_back(slot.flight->start);
            requests.segmentSlots.push_back(pending.id); requests.segments.push_back(requests.firstSegments.back());
            slot.stage = SlotStage::SegmentInFlight;
            d.maximumInitializationWait = std::max(d.maximumInitializationWait,round-pending.round);
        }
        for (std::size_t i=0;i<ns;++i) {
            const int id = segmentQueue.front().id; segmentQueue.pop_front();
            auto& slot = slots[id];
            if (slot.stage != SlotStage::Segment) throw std::logic_error("invalid Renewal segment queue");
            requests.segmentSlots.push_back(id); requests.segments.push_back(slot.flight->mean.segment(slot.segment));
            slot.stage = SlotStage::SegmentInFlight;
        }
        return true;
    };
    const auto recordSubmission = [&](const RenewalBatchRequests& requests) {
        const auto ni=requests.initializeSlots.size(), ns=requests.segmentSlots.size(), nm=requests.mixtureSlots.size();
        ++d.submissions;
        if (static_cast<int>(ni>0)+static_cast<int>(ns>0)+static_cast<int>(nm>0)>1) ++d.combinedSubmissions;
        if (ni) ++d.initializationBatches;
        if (ns) ++d.segmentBatches;
        if (nm) ++d.mixtureBatches;
        d.segments += ns; d.mixtureQueries += nm;
        d.maximumBatchSize = std::max(d.maximumBatchSize,static_cast<std::uint64_t>(ns));
        d.maximumInitializationBatch = std::max(d.maximumInitializationBatch,static_cast<std::uint64_t>(ni));
        d.maximumMixtureBatch = std::max(d.maximumMixtureBatch,static_cast<std::uint64_t>(nm));
    };
    const auto collectBatch = [&](const std::shared_ptr<BatchWork>& batch) {
        --outstandingInference;
        ++d.readbacks; ++round;
        if (batch->otherGpuPending) ++d.cpuBatchesWithGpuPending;
        const auto ns = batch->requests.segmentSlots.size();
        if (ns) {
            if (cpu.workerCount()>1 && ns>RenewalCpuExecutor::grain) ++d.parallelAdvanceBatches;
            else ++d.serialAdvanceBatches;
        }
        // CPU tasks retain request/results after the GPU transfer frame is freed.
        // Successors are published chunk by chunk, with no batch-wide barrier.
        enqueueRanges(ns,[&,batch](std::size_t begin,std::size_t end,std::vector<int>& ready) {
            for (auto i=begin;i<end;++i) {
                const int id = batch->requests.segmentSlots[i];
                advanceSlot(slots[id],batch->requests.segments[i],batch->output.rates[i]);
                if (slots[id].stage != SlotStage::Retired) ready.push_back(id);
            }
        });
        enqueueRanges(batch->requests.mixtureSlots.size(),[&,batch](std::size_t begin,std::size_t end,std::vector<int>& ready) {
            for (auto i=begin;i<end;++i) {
                const int id = batch->requests.mixtureSlots[i];
                scatterSlot(slots[id],batch->requests.coordinates[i],batch->output.mixtures[i]);
                if (slots[id].stage != SlotStage::Retired) ready.push_back(id);
            }
        });
    };
    const auto inferenceLoop = [&]() {
        try {
            RenewalBatchSession network(*config.renewal.model,capacity,config.renewal.resolvedBackend,pipelineDepth);
            std::deque<std::shared_ptr<BatchWork>> pendingBatches;
            for (;;) {
                { std::lock_guard<std::mutex> lock(readyMutex); if (stoppingInference) break; }
                // Publish this batch before entering the next long ATen call,
                // even if its short device tail still needs an event wait.
                // Otherwise completion can be hidden behind the next dispatch,
                // leaving CPU workers idle throughout that GPU batch as well.
                const bool resultReady = !pendingBatches.empty() && network.isReady(pendingBatches.front()->ticket);
                std::shared_ptr<BatchWork> batch;
                {
                    std::lock_guard<std::mutex> lock(readyMutex);
                    if (pendingBatches.empty() && !inferenceQueue.empty()) {
                        batch = inferenceQueue.front(); inferenceQueue.pop_front();
                    }
                }
                if (batch) {
                    batch->ticket = network.submitWavefrontAsync(batch->requests,preparedFeatures);
                    pendingBatches.push_back(batch);
                    maximumGpuPending = std::max(maximumGpuPending,static_cast<std::uint64_t>(pendingBatches.size()));
                    continue;
                }
                if (!pendingBatches.empty()) {
                    batch = pendingBatches.front(); pendingBatches.pop_front();
                    if (!resultReady) ++blockingCollects;
                    // Only this inference thread waits for the CUDA event.
                    // Avoid timed polling: OS timer granularity can be much
                    // longer than these short kernels. CPU workers and their
                    // coordinator continue independently during the event wait.
                    network.collectInto(batch->ticket,batch->output);
                    batch->otherGpuPending = !pendingBatches.empty() && !network.isReady(pendingBatches.front()->ticket);
                    { std::lock_guard<std::mutex> lock(readyMutex); completedInference.push_back(batch); }
                    readyCondition.notify_one();
                    continue;
                }
                std::unique_lock<std::mutex> lock(readyMutex);
                const auto ready = [&] { return stoppingInference || !inferenceQueue.empty(); };
                inferenceCondition.wait(lock,ready);
            }
        } catch (...) {
            { std::lock_guard<std::mutex> lock(readyMutex);
              if (!failure) failure = std::current_exception();
              cancelled.store(true,std::memory_order_relaxed); }
            readyCondition.notify_one();
        }
    };
    const auto stopInference = [&]() {
        { std::lock_guard<std::mutex> lock(readyMutex); stoppingInference = true; }
        inferenceCondition.notify_one();
        if (inferenceThread.joinable()) inferenceThread.join();
    };
    try {
        inferenceThread = std::thread(inferenceLoop);
        enqueueRanges(static_cast<std::size_t>(capacity),[&](std::size_t begin,std::size_t end,std::vector<int>& ready) {
            for (auto i=begin;i<end;++i) {
                initializeSlot(slots[i]);
                if (slots[i].stage != SlotStage::Retired) ready.push_back(static_cast<int>(i));
            }
        });
        for (;;) {
            const auto activeCpu = drainIncoming();
            if (!receivedInference.empty()) {
                for (const auto& batch : receivedInference) collectBatch(batch);
                receivedInference.clear();
                continue; // CPU task count changed; refresh the snapshot.
            }
            if (outstandingInference < static_cast<std::size_t>(pipelineDepth)) {
                auto batch = batchPool.acquire();
                if (prepareBatch(batch->requests,activeCpu)) {
                    { std::lock_guard<std::mutex> lock(readyMutex); inferenceQueue.push_back(batch); }
                    ++outstandingInference; recordSubmission(batch->requests);
                    inferenceCondition.notify_one();
                    continue;
                }
            }
            if (!activeCpu && !outstandingInference) {
                if (!initializationQueue.empty() || !segmentQueue.empty() || !mixtureQueue.empty())
                    throw std::logic_error("Renewal ready queue failed to drain");
                break;
            }
            // Completion of either CPU work or inference wakes the coordinator.
            std::unique_lock<std::mutex> lock(readyMutex);
            const auto ready = [&] { return failure || !incoming.empty() || !completedInference.empty() ||
                (activeCpu && !outstandingCpuTasks); };
            readyCondition.wait(lock,ready);
        }
        cpu.wait();
        stopInference();
        { std::lock_guard<std::mutex> lock(readyMutex); if (failure) std::rethrow_exception(failure); }
    } catch (...) {
        const auto error = std::current_exception();
        cancelled.store(true,std::memory_order_relaxed);
        // Keep callbacks, slots and batch buffers alive until every worker exits.
        // The session then drains outstanding GPU transfers during destruction.
        try { cpu.wait(); } catch (...) {}
        stopInference();
        std::rethrow_exception(error);
    }
    for (const auto& slot : slots) {
        if (slot.stage != SlotStage::Retired) throw std::logic_error("unfinished Renewal slot");
        mergeInto(statistics,slot.statistics);
    }
    d.backend = config.renewal.resolvedBackend;
    d.maximumInFlightBatches = maximumGpuPending;
    d.blockingCollects = blockingCollects;
    return image;
}
} // namespace mf

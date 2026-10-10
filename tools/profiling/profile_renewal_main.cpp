#include "RenewalProfile.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/integrator/RenewalWavefront.h"
#include <ATen/Parallel.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) {
        std::cerr << "usage: macrofacet_profile_renewal config spp repeats output.json baseline|wall|trace [point_query_stride=64] [torch_cpu_threads=0]\n";
        return 2;
    }
    try {
        mf::registerNanoVdbFieldTypes();
        auto config = mf::loadExperimentConfig(argv[1]);
        config.render.samplesPerPixel = std::stoi(argv[2]);
        const int repeats = std::stoi(argv[3]);
        const std::string mode = argv[5];
        if (argc >= 7) {
            const int stride = std::stoi(argv[6]);
            if (stride < 1) throw std::invalid_argument("point query stride must be positive");
            renewal_profile::pointQueryStride = static_cast<std::uint64_t>(stride);
        }
        const int torchThreads = argc == 8 ? std::stoi(argv[7]) : 0;
        if (torchThreads < 0) throw std::invalid_argument("torch CPU threads must be nonnegative");
        if (torchThreads) at::set_num_threads(torchThreads);
        if (config.render.samplesPerPixel < 1 || repeats < 1 ||
            (mode != "baseline" && mode != "wall" && mode != "trace"))
            throw std::invalid_argument("invalid profiling arguments");
        mf::prepareRenewalModel(config);
        const bool cuda = config.renewal.resolvedBackend == "torch_cuda";
        if (!cuda && config.renewal.resolvedBackend != "torch_cpu")
            throw std::invalid_argument("wavefront profiling requires torch_cpu, torch_cuda or auto backend");
        if (mode == "trace" && !cuda)
            throw std::invalid_argument("CUDA trace mode requires torch_cuda backend");
        mf::prepareNanoVdbField(config);
        if (config.render.environment == "unit_white") config.material.conductor.forceUnitFresnel = true;
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output.parent_path());
        auto warm = config;
        warm.render.samplesPerPixel = 1;
        std::cerr << "warming " << config.renewal.resolvedBackend << " and renderer; Torch CPU threads: "
                  << at::get_num_threads() << ", interop: " << at::get_num_interop_threads() << '\n';
        mf::renderRenewalWavefront(warm,nullptr);
        cudaDeviceProp device{};
        if (cuda) {
            cudaDeviceSynchronize();
            cudaGetDeviceProperties(&device,0);
        }
        nlohmann::json report{{"config",argv[1]}, {"mode",mode}, {"gpu",device.name},
            {"width",config.render.width}, {"height",config.render.height},
            {"spp",config.render.samplesPerPixel}, {"batch_capacity",config.renewal.batchSize},
            {"profile_mode",config.renewal.profileMode},
            {"point_query_stride",renewal_profile::pointQueryStride},
            {"backend",config.renewal.resolvedBackend},
            {"torch_cpu_threads_requested",torchThreads},
            {"torch_cpu_threads",at::get_num_threads()},
            {"torch_interop_threads",at::get_num_interop_threads()},
            {"checkpoint_sha256",config.renewal.model->checkpointSha256()},
            {"note","Warm render. Field/model loading and warmup excluded. Host timers include waits; async module host times are not GPU execution times. No extra per-stage CUDA synchronization."},
            {"runs",nlohmann::json::array()}};
        for (int run = 0; run < repeats; ++run) {
            renewal_profile::reset();
            renewal_profile::enabled = mode != "baseline";
            if (mode == "trace") cudaProfilerStart();
            const auto start = renewal_profile::Clock::now();
            auto image = renewal_profile::measure("render.measured",[&]() {
                return mf::renderRenewalWavefront(config,nullptr);
            });
            const double seconds = std::chrono::duration<double>(renewal_profile::Clock::now()-start).count();
            if (mode == "trace") cudaProfilerStop();
            renewal_profile::enabled = false;
            const auto& s = image.statistics;
            const auto measurements = renewal_profile::snapshot();
            nlohmann::json stages;
            for (const auto& [name,value] : measurements.timings)
                stages[name] = {{"inclusive_seconds",value.inclusive}, {"exclusive_seconds",value.exclusive}, {"calls",value.calls}};
            nlohmann::json workerStages;
            for (const auto& [name,value] : measurements.workerTimings)
                workerStages[name] = {{"inclusive_seconds",value.inclusive}, {"exclusive_seconds",value.exclusive}, {"calls",value.calls}};
            nlohmann::json result{{"seconds",seconds}, {"stages",stages}, {"worker_stages",workerStages}, {"counts",measurements.counts},
                {"point_queries",measurements.pointQueryCalls},
                {"worker_point_queries",measurements.workerPointQueryCalls},
                {"point_query_sampled_thread_seconds",measurements.sampledQueries.inclusive},
                {"point_query_sampled_calls",measurements.sampledQueries.calls},
                {"cpu_workers",s.renewal.cpuWorkers},
                {"parallel_advance_batches",s.renewal.parallelAdvanceBatches},
                {"serial_advance_batches",s.renewal.serialAdvanceBatches},
                {"submissions",s.renewal.submissions},
                {"readbacks",s.renewal.readbacks},
                {"max_in_flight_batches",s.renewal.maximumInFlightBatches},
                {"blocking_collects",s.renewal.blockingCollects},
                {"cpu_batches_with_gpu_pending",s.renewal.cpuBatchesWithGpuPending},
                {"combined_submissions",s.renewal.combinedSubmissions},
                {"ray_pool_size",s.renewal.rayPoolSize},
                {"max_initialization_batch",s.renewal.maximumInitializationBatch},
                {"max_mixture_batch",s.renewal.maximumMixtureBatch},
                {"max_initialization_wait",s.renewal.maximumInitializationWait},
                {"max_mixture_wait",s.renewal.maximumMixtureWait},
                {"paths",s.paths}, {"flights",s.renewal.flights}, {"segments",s.renewal.segments},
                {"segment_batches",s.renewal.segmentBatches}, {"initialization_batches",s.renewal.initializationBatches},
                {"mixture_batches",s.renewal.mixtureBatches}, {"hits",s.realCollisions},
                {"numerical_failures",s.numericalFailures}, {"safety_cap_terminations",s.safetyCapTerminations},
                {"average_segment_batch",double(s.renewal.segments)/s.renewal.segmentBatches}};
            report["runs"].push_back(result);
            std::ofstream(output) << report.dump(2) << '\n';
            if (run == 0) mf::writePfm(output.string()+".pfm",image);
            std::cerr << "run " << run+1 << ": " << seconds << " s, " << s.renewal.segments
                      << " segments / " << s.renewal.segmentBatches << " batches\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}

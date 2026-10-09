#include "RenewalProfile.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "macrofacet/fields/PrepareNanoVdbField.h"
#include "macrofacet/integrator/RenewalWavefront.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: macrofacet_profile_renewal config spp repeats output.json baseline|wall|trace [point_query_stride=64]\n";
        return 2;
    }
    try {
        mf::registerNanoVdbFieldTypes();
        auto config = mf::loadExperimentConfig(argv[1]);
        config.render.samplesPerPixel = std::stoi(argv[2]);
        const int repeats = std::stoi(argv[3]);
        const std::string mode = argv[5];
        if (argc == 7) {
            const int stride = std::stoi(argv[6]);
            if (stride < 1) throw std::invalid_argument("point query stride must be positive");
            renewal_profile::pointQueryStride = static_cast<std::uint64_t>(stride);
        }
        if (config.render.samplesPerPixel < 1 || repeats < 1 ||
            (mode != "baseline" && mode != "wall" && mode != "trace"))
            throw std::invalid_argument("invalid profiling arguments");
        config.renewal.backend = "torch_cuda";
        mf::prepareRenewalModel(config);
        mf::prepareNanoVdbField(config);
        if (config.render.environment == "unit_white") config.material.conductor.forceUnitFresnel = true;
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output.parent_path());
        auto warm = config;
        warm.render.samplesPerPixel = 1;
        std::cerr << "warming CUDA and renderer\n";
        mf::renderRenewalWavefront(warm,nullptr);
        cudaDeviceSynchronize();
        cudaDeviceProp device{};
        cudaGetDeviceProperties(&device,0);
        nlohmann::json report{{"config",argv[1]}, {"mode",mode}, {"gpu",device.name},
            {"width",config.render.width}, {"height",config.render.height},
            {"spp",config.render.samplesPerPixel}, {"batch_capacity",config.renewal.batchSize},
            {"profile_mode",config.renewal.profileMode},
            {"point_query_stride",renewal_profile::pointQueryStride},
            {"backend",config.renewal.resolvedBackend},
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

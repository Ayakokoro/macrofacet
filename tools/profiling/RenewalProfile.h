#pragma once
// Diagnostic executable only. No instrumentation is compiled into the renderer.
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace renewal_profile {
using Clock = std::chrono::steady_clock;
struct Measurement { double inclusive = 0, exclusive = 0; std::uint64_t calls = 0; };
inline bool enabled = false;
// A timer/NVTX range per voxel substantially perturbs the traversal. Sample
// queries instead; the enclosing batch timers still measure the complete work.
inline std::uint64_t pointQueryStride = 64;
struct ThreadRecord {
    std::map<std::string, Measurement> timings;
    std::map<std::string, std::uint64_t> counts;
    std::uint64_t pointQueryCalls = 0;
    bool coordinator = false;
};
inline std::mutex recordsMutex;
inline std::vector<std::shared_ptr<ThreadRecord>> records;
inline std::thread::id coordinatorId;
inline thread_local std::shared_ptr<ThreadRecord> threadRecord;
inline ThreadRecord& local() {
    if (!threadRecord) {
        threadRecord = std::make_shared<ThreadRecord>();
        threadRecord->coordinator = std::this_thread::get_id() == coordinatorId;
        std::lock_guard<std::mutex> lock(recordsMutex);
        records.push_back(threadRecord);
    }
    return *threadRecord;
}
// Only between frames, after all renderer workers have joined.
inline void reset() {
    enabled = false;
    coordinatorId = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(recordsMutex);
    records.erase(std::remove_if(records.begin(),records.end(),
        [](const auto& r) { return r.use_count() == 1; }),records.end());
    for (auto& r : records) {
        r->timings.clear(); r->counts.clear(); r->pointQueryCalls = 0;
    }
}
struct Snapshot {
    std::map<std::string, Measurement> timings, workerTimings;
    std::map<std::string, std::uint64_t> counts;
    std::uint64_t pointQueryCalls = 0, workerPointQueryCalls = 0;
    Measurement sampledQueries;
};
inline Snapshot snapshot() {
    Snapshot result;
    std::lock_guard<std::mutex> lock(recordsMutex);
    for (const auto& r : records) {
        auto& timings = r->coordinator ? result.timings : result.workerTimings;
        for (const auto& [name,m] : r->timings) {
            auto& sum = timings[name];
            sum.inclusive += m.inclusive; sum.exclusive += m.exclusive; sum.calls += m.calls;
            if (name == "profile.point_query_sampled") {
                result.sampledQueries.inclusive += m.inclusive;
                result.sampledQueries.calls += m.calls;
            }
        }
        for (const auto& [name,n] : r->counts) result.counts[name] += n;
        result.pointQueryCalls += r->pointQueryCalls;
        if (!r->coordinator) result.workerPointQueryCalls += r->pointQueryCalls;
    }
    return result;
}
struct Scope;
inline thread_local Scope* current = nullptr;
struct Scope {
    const char* name;
    Scope* parent = nullptr;
    Clock::time_point start;
    double children = 0;
    bool active;
    ThreadRecord* record = nullptr;
    explicit Scope(const char* label) : name(label), active(enabled) {
        if (!active) return;
        record = &local();
        parent = current; current = this;
        nvtxRangePushA(name);
        start = Clock::now();
    }
    ~Scope() {
        if (!active) return;
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        auto& value = record->timings[name];
        value.inclusive += seconds; value.exclusive += seconds-children; ++value.calls;
        if (parent) parent->children += seconds;
        current = parent;
        nvtxRangePop();
    }
};
template<class F> decltype(auto) measure(const char* name, F&& function) {
    Scope scope(name);
    return std::forward<F>(function)();
}
template<class F> decltype(auto) measurePointQuery(F&& function) {
    if (!enabled || (++local().pointQueryCalls-1)%pointQueryStride != 0)
        return std::forward<F>(function)();
    Scope scope("profile.point_query_sampled");
    return std::forward<F>(function)();
}
inline void count(const char* name, std::uint64_t value) {
    if (enabled) local().counts[name] += value;
}
}

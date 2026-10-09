#pragma once
// Diagnostic executable only. No instrumentation is compiled into the renderer.
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace renewal_profile {
using Clock = std::chrono::steady_clock;
struct Measurement { double inclusive = 0, exclusive = 0; std::uint64_t calls = 0; };
inline bool enabled = false;
// A timer/NVTX range per voxel substantially perturbs the traversal. Sample
// queries instead; the enclosing batch timers still measure the complete work.
inline std::uint64_t pointQueryStride = 64, pointQueryCalls = 0;
inline std::map<std::string, Measurement> timings;
inline std::map<std::string, std::uint64_t> counts;
struct Scope;
inline thread_local Scope* current = nullptr;
struct Scope {
    const char* name;
    Scope* parent = nullptr;
    Clock::time_point start;
    double children = 0;
    bool active;
    explicit Scope(const char* label) : name(label), active(enabled) {
        if (!active) return;
        parent = current; current = this;
        nvtxRangePushA(name);
        start = Clock::now();
    }
    ~Scope() {
        if (!active) return;
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        auto& value = timings[name];
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
    if (!enabled || (++pointQueryCalls-1)%pointQueryStride != 0)
        return std::forward<F>(function)();
    Scope scope("profile.point_query_sampled");
    return std::forward<F>(function)();
}
inline void count(const char* name, std::uint64_t value) {
    if (enabled) counts[name] += value;
}
}

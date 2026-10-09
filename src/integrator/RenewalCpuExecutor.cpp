#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include <algorithm>
#include <stdexcept>

namespace mf {
RenewalCpuExecutor::RenewalCpuExecutor(int requested, std::size_t capacity) {
    if (requested < 0 || capacity == 0) throw std::invalid_argument("invalid Renewal CPU pool settings");
    const auto available = std::max(1u,std::thread::hardware_concurrency());
    // Automatic mode is deliberately conservative for the short per-batch work.
    const auto desired = requested ? static_cast<std::size_t>(requested) : std::min(8u,available);
    const auto workers = capacity < parallelThreshold ? 1 :
        std::max<std::size_t>(1,std::min(desired,capacity/grain));
    try {
        for (std::size_t i = 0; i+1 < workers; ++i)
            threads_.emplace_back([this,i] { workerLoop(i); });
    } catch (...) { stop(); throw; }
}
RenewalCpuExecutor::~RenewalCpuExecutor() { stop(); }

void RenewalCpuExecutor::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_all();
    for (auto& thread : threads_) if (thread.joinable()) thread.join();
}

void RenewalCpuExecutor::executeRanges() {
    try {
        while (!cancelled_.load(std::memory_order_relaxed)) {
            const auto begin = next_.fetch_add(grain,std::memory_order_relaxed);
            if (begin >= count_) break;
            function_(begin,std::min(count_,begin+grain));
        }
    } catch (...) {
        cancelled_.store(true,std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_) failure_ = std::current_exception();
    }
}

void RenewalCpuExecutor::workerLoop(std::size_t worker) {
    std::size_t observed = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        ready_.wait(lock,[&] { return stopping_ || generation_ != observed; });
        if (stopping_) return;
        observed = generation_;
        if (worker >= participating_) continue;
        lock.unlock();
        executeRanges();
        lock.lock();
        if (--pending_ == 0) done_.notify_one();
    }
}

bool RenewalCpuExecutor::forRanges(std::size_t count,
    const std::function<void(std::size_t,std::size_t)>& function) {
    if (count == 0) return false;
    if (threads_.empty() || count < parallelThreshold) {
        function(0,count);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        function_ = function;
        count_ = count; next_.store(0,std::memory_order_relaxed);
        cancelled_.store(false,std::memory_order_relaxed); failure_ = nullptr;
        participating_ = std::min(threads_.size(),(count+grain-1)/grain-1);
        pending_ = participating_;
        ++generation_;
    }
    ready_.notify_all();
    executeRanges();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock,[&] { return pending_ == 0; });
    function_ = {};
    const auto failure = failure_;
    lock.unlock();
    if (failure) std::rethrow_exception(failure);
    return true;
}
} // namespace mf

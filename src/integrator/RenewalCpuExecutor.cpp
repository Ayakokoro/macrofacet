#include "macrofacet/integrator/RenewalCpuExecutor.h"
#include <algorithm>
#include <stdexcept>

namespace mf {
RenewalCpuExecutor::RenewalCpuExecutor(int requested, std::size_t capacity) {
    if (requested < 0 || capacity == 0) throw std::invalid_argument("invalid Renewal CPU pool settings");
    const auto available = std::max(1u,std::thread::hardware_concurrency());
    const auto desired = requested ? static_cast<std::size_t>(requested) : std::min(8u,available);
    const auto workers = std::max<std::size_t>(1,std::min(desired,(capacity+grain-1)/grain));
    try {
        for (std::size_t i = 0; i < workers; ++i) threads_.emplace_back([this] { workerLoop(); });
    } catch (...) { stop(); throw; }
}
RenewalCpuExecutor::~RenewalCpuExecutor() { stop(); }

void RenewalCpuExecutor::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_all();
    for (auto& thread : threads_) if (thread.joinable()) thread.join();
}

void RenewalCpuExecutor::enqueue(std::function<void()> function) {
    if (!function) throw std::invalid_argument("empty Renewal CPU task");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) throw std::logic_error("Renewal CPU executor stopped");
        tasks_.push_back(std::move(function));
    }
    ready_.notify_one();
}
void RenewalCpuExecutor::workerLoop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock,[&] { return stopping_ || !tasks_.empty(); });
            if (tasks_.empty()) return;
            task = std::move(tasks_.front()); tasks_.pop_front(); ++active_;
        }
        try { task(); }
        catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
        task = {}; // Release captured batch buffers before waking waiters.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_;
            if (tasks_.empty() && !active_) done_.notify_all();
        }
    }
}
void RenewalCpuExecutor::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock,[&] { return tasks_.empty() && !active_; });
    const auto failure = failure_; failure_ = nullptr;
    lock.unlock();
    if (failure) std::rethrow_exception(failure);
}
bool RenewalCpuExecutor::forRanges(std::size_t count,
    const std::function<void(std::size_t,std::size_t)>& function) {
    if (!count) return false;
    const bool parallel = workerCount() > 1 && count >= parallelThreshold;
    const auto step = parallel ? grain : count;
    try {
        for (std::size_t begin = 0; begin < count; begin += step)
            enqueue([=] { function(begin,std::min(count,begin+step)); });
    } catch (...) { const auto failure=std::current_exception(); wait(); std::rethrow_exception(failure); }
    wait();
    return parallel;
}
} // namespace mf

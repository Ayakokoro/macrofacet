#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mf {
// Persistent background workers; neither scheduler nor inference thread executes
// ray work. Callbacks own their data until completion. enqueue never waits for
// another task or a batch; wait drains work and propagates callback failures.
class RenewalCpuExecutor {
public:
    static constexpr std::size_t grain = 128;
    static constexpr std::size_t parallelThreshold = 512;
    RenewalCpuExecutor(int requestedThreads, std::size_t capacity);
    ~RenewalCpuExecutor();
    RenewalCpuExecutor(const RenewalCpuExecutor&) = delete;
    RenewalCpuExecutor& operator=(const RenewalCpuExecutor&) = delete;
    int workerCount() const { return static_cast<int>(threads_.size()); }
    void enqueue(std::function<void()> function);
    void wait();
    // Compatibility wrapper for callers that explicitly need a barrier.
    bool forRanges(std::size_t count, const std::function<void(std::size_t,std::size_t)>& function);
private:
    void workerLoop();
    void stop();
    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    bool stopping_ = false;
    std::size_t active_ = 0;
    std::exception_ptr failure_;
};
} // namespace mf

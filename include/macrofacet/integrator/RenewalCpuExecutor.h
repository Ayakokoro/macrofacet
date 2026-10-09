#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mf {
// Synchronous range dispatch to a persistent pool. The calling thread participates.
// One coordinator owns the executor; callbacks must own disjoint output ranges.
class RenewalCpuExecutor {
public:
    static constexpr std::size_t grain = 128;
    static constexpr std::size_t parallelThreshold = 512;
    RenewalCpuExecutor(int requestedThreads, std::size_t capacity);
    ~RenewalCpuExecutor();
    RenewalCpuExecutor(const RenewalCpuExecutor&) = delete;
    RenewalCpuExecutor& operator=(const RenewalCpuExecutor&) = delete;
    int workerCount() const { return static_cast<int>(threads_.size())+1; }
    // Returns true if the pool was dispatched. All callbacks finish before return
    // or rethrow, including on failure; the pool can then be reused.
    bool forRanges(std::size_t count, const std::function<void(std::size_t,std::size_t)>& function);
private:
    void workerLoop(std::size_t worker);
    void executeRanges();
    void stop();
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    bool stopping_ = false;
    std::size_t generation_ = 0, participating_ = 0, pending_ = 0, count_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<bool> cancelled_{false};
    std::function<void(std::size_t,std::size_t)> function_;
    std::exception_ptr failure_;
};
} // namespace mf

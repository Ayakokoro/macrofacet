#pragma once

namespace mf {

// Keep the low part of a running sum, including positive increments smaller
// than one ulp of its current value. Uses two doubles also on MSVC, where
// long double has the same precision as double.
class CompensatedSum {
public:
    explicit CompensatedSum(double value = 0.0) : high_(value) {}
    double value() const { return high_; }
    void add(double increment) {
        const double sum = high_ + increment;
        const double moved = sum - high_;
        const double error = (high_ - (sum - moved)) + (increment - moved);
        const double tail = low_ + error;
        const double combined = sum + tail;
        low_ = tail - (combined - sum);
        high_ = combined;
    }
private:
    double high_, low_ = 0.0;
};

} // namespace mf

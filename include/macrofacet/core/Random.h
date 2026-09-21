#pragma once

#include "macrofacet/core/Types.h"
#include <cstdint>
#include <limits>
#include <random>

namespace mf {

class Random {
public:
    explicit Random(std::uint64_t seed = 1) : engine_(seed) {}

    double uniform01() {
        return std::generate_canonical<double, 64>(engine_);
    }

    double openUniform01() {
        double u = 0.0;
        do { u = uniform01(); } while (!(u > 0.0 && u < 1.0));
        return u;
    }

    double standardNormal() { return normal_(engine_); }
    std::uint64_t integer() { return engine_(); }

private:
    std::mt19937_64 engine_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};

} // namespace mf


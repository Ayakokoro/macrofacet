#pragma once
// Plain data interface: no Eigen, LibTorch, or host renderer objects in kernels.
#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>

namespace mf::cuda_renewal {
struct V3 { double x, y, z; };
struct Dense { int input, output, offset, bias; };
struct Network {
    Dense initial0, initial2, segment0, segment2, head0, head2;
    Dense gruInput, gruHidden, mixture0, mixture2, mixture4;
    int hidden, embedding, components;
    double step, fine;
};
struct Scene {
    V3 minimum, maximum, camera, forward, right, up, eta, k;
    double sigma, ell, profileStep, tolerance, aspect, cameraScale;
    int width, height, spp, rouletteDepth, safetyDepth, white, unitFresnel, deltaTracking;
    std::uint64_t seed;
};
struct Input {
    V3 origin, direction, birthGradient;
    double uniforms[3], normals[2];
    int surface, queryOnly;
    std::uint64_t trackingSeed;
};
struct Output {
    V3 gradient;
    double q, speed, hazard;
    std::uint64_t intervals;
    int hit, error;
    std::uint64_t candidates, nulls, bounds;
};
struct Statistics {
    unsigned long long completed, collisions, escaped, roulette, capped, errors;
    unsigned long long depth, flights, intervals, kernels;
    unsigned long long candidates, nulls, bounds;
};
class Context {
public:
    Context(const void* grid, std::size_t bytes, const Network&, const std::vector<float>&, const Scene&);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    std::vector<Output> sample(const std::vector<Input>&);
    std::vector<V3> render(int capacity, Statistics&, const std::function<void(std::uint64_t)>&);
private:
    struct Impl;
    Impl* impl_;
};
} // namespace mf::cuda_renewal

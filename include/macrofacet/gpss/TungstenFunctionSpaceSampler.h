#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace mf {

inline constexpr const char* kTungstenFunctionSpaceMethod = "gpis_function_space_tungsten";

// The supported Tungsten variant is standard SE, renewal+, gp_discrete,
// conditioned_gaussian, skip_space=0. stepSize is a WORLD-space distance.
struct TungstenFunctionSpaceConfig {
    int samplePoints = 8;
    double stepSize = 0.01; // 0: one batch spanning the remaining ray
    void validate() const;
};

// Tungsten UniformSampler (PCG + 23-bit float uniforms). A supplied stream is
// useful for replaying exactly the same PathSampleGenerator inputs in tests.
class TungstenRandom {
public:
    explicit TungstenRandom(std::uint64_t seed, std::uint64_t sequence = 0);
    explicit TungstenRandom(std::function<double()> uniformSource);
    double uniform01();
    Vector2 normalPair();
    Eigen::VectorXd normals(int count);
    double positiveStart(double mean, double sigma);
    std::uint64_t draws() const { return draws_; }
private:
    std::uint64_t state_ = 0, sequence_ = 0, draws_ = 0;
    std::function<double()> source_;
};

struct TungstenObservation {
    Point3 position = Point3::Zero();
    bool derivative = false;
    Vector3 direction = Vector3::Zero();
};

// Keep the GP observation position separate from the float scatter position.
// The interpolated value is approximately zero at a hit, not forcibly rounded.
struct TungstenSurfaceState {
    Point3 position = Point3::Zero();
    double value = 0;
    Vector3 gradient = Vector3::Zero();
};

struct TungstenGaussian {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

struct TungstenFunctionSpaceBatch {
    std::vector<double> times;
    std::vector<TungstenObservation> observations;
    Eigen::VectorXd values;
    double distance = 0;
    bool hit = false;
    int initialSign = 1;
    TungstenSurfaceState end;
};

class TungstenFunctionSpaceSampler {
public:
    TungstenFunctionSpaceSampler(const GPSSField& field, TungstenFunctionSpaceConfig config);
    const TungstenFunctionSpaceConfig& config() const { return config_; }

    TungstenGaussian conditional(const std::vector<TungstenObservation>& query,
        const std::vector<TungstenObservation>& observations,
        const Eigen::VectorXd& values) const;
    static Eigen::VectorXd draw(const TungstenGaussian& gaussian, TungstenRandom& rng);

    // Ray origin/direction and near/far have Tungsten float precision. The
    // direction is normalized in double for point placement, as in intersectGP.
    TungstenFunctionSpaceBatch sampleBatch(const Ray& ray, float nearT, float farT,
        const std::optional<TungstenSurfaceState>& previous, bool firstIntersectionAlongRay,
        TungstenRandom& rng) const;

    // Factored out to regression-test the original skipped interval and secant
    // context construction using explicitly prescribed function values.
    static void locateIntersection(TungstenFunctionSpaceBatch& batch,
        const Ray& ray, bool skipFirstInterval);
private:
    const GPSSField& field_;
    TungstenFunctionSpaceConfig config_;
    double covariance(const TungstenObservation& a, const TungstenObservation& b) const;
    Eigen::VectorXd mean(const std::vector<TungstenObservation>& observations) const;
    Vector3 sampleGradient(const TungstenFunctionSpaceBatch& batch,
        const Vector3& direction, TungstenRandom& rng) const;
};

} // namespace mf

#pragma once

#include "macrofacet/core/Types.h"
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mf {

struct FirstPassageMlpConfig {
    std::string type = "mlp";
    std::filesystem::path modulePath;
    std::filesystem::path bundlePath;
    std::string device = "cpu";
};

struct FirstPassageMlpMetadata {
    std::string format;
    int formatVersion = 0;
    std::string kernelId;
    std::string kernelType;
    int basisDegree = 0;
    int basisCount = 0;
    double maximumQ = 0.0;
    std::vector<double> knots;
    Vector3 validMinimum = Vector3::Zero();
    Vector3 validMaximum = Vector3::Zero();
};

// Kernel-independent TorchScript coefficient model. The module owns the input
// transform and neural network; the JSON bundle defines only the common
// monotone I-spline representation used by the renderer.
class FirstPassageMlp {
public:
    explicit FirstPassageMlp(FirstPassageMlpConfig config);
    ~FirstPassageMlp();
    FirstPassageMlp(FirstPassageMlp&&) noexcept;
    FirstPassageMlp& operator=(FirstPassageMlp&&) noexcept;
    FirstPassageMlp(const FirstPassageMlp&) = delete;
    FirstPassageMlp& operator=(const FirstPassageMlp&) = delete;

    const FirstPassageMlpConfig& config() const { return config_; }
    const FirstPassageMlpMetadata& metadata() const { return metadata_; }

    std::vector<double> coefficients(const Vector3& beta) const;
    bool contains(const Vector3& beta, double q) const;
    std::vector<double> mSpline(double q) const;
    std::vector<double> iSpline(double q) const;
    double cumulativeHazard(const std::vector<double>& coefficients, double q) const;
    double transmittance(const std::vector<double>& coefficients, double q) const;
    double dimensionlessExtinction(const std::vector<double>& coefficients, double q) const;
    double inverseCumulativeHazard(const std::vector<double>& coefficients,
                                   double opticalDepth,
                                   int iterations = 64) const;

private:
    struct Impl;
    FirstPassageMlpConfig config_;
    FirstPassageMlpMetadata metadata_;
    std::unique_ptr<Impl> impl_;
};

} // namespace mf

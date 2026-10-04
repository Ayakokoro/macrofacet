#include "TestHarness.h"
#include "macrofacet/learned/FirstPassageMlp.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

void testFirstPassageMlp(TestContext& context) {
    using namespace mf;
    namespace fs = std::filesystem;
    const fs::path directory = MACROFACET_MLP_TEST_DATA;
    const FirstPassageMlpConfig config{
        "mlp", directory / "coefficient_net.pt", directory / "model_bundle.json", "cpu"};
    FirstPassageMlp model(config);
    context.require(model.metadata().kernelType == "fixture_kernel_without_cpp_dispatch",
                    "MLP loader treats kernel type as metadata rather than C++ dispatch");
    context.require(model.metadata().basisCount == 4 && model.metadata().basisDegree == 2,
                    "MLP loader reads the common spline representation");

    nlohmann::json golden;
    {
        std::ifstream stream(directory / "golden.json");
        stream >> golden;
    }
    const Vector3 beta(golden["beta"][0].get<double>(),
                       golden["beta"][1].get<double>(),
                       golden["beta"][2].get<double>());
    const std::vector<double> coefficients = model.coefficients(beta);
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        context.near(coefficients[index], golden["coefficients"][index].get<double>(), 1e-12,
                     "LibTorch coefficient inference matches Python");
    }
    context.require(model.contains(beta, 1.0) && !model.contains(beta, 3.0) &&
                    !model.contains(Vector3(1.0, 2.0, 0.01), 1.0),
                    "MLP validity box checks beta and q");

    for (std::size_t sample = 0; sample < golden["q"].size(); ++sample) {
        const double q = golden["q"][sample].get<double>();
        const std::vector<double> iValues = model.iSpline(q);
        const std::vector<double> mValues = model.mSpline(q);
        for (std::size_t basis = 0; basis < coefficients.size(); ++basis) {
            context.near(iValues[basis], golden["i_spline"][sample][basis].get<double>(),
                         2e-12, "C++ I-spline matches SciPy");
            context.near(mValues[basis], golden["m_spline"][sample][basis].get<double>(),
                         2e-12, "C++ M-spline matches SciPy");
        }
        context.near(model.cumulativeHazard(coefficients, q),
                     golden["cumulative_hazard"][sample].get<double>(), 2e-11,
                     "C++ cumulative hazard matches Python");
        context.near(model.dimensionlessExtinction(coefficients, q),
                     golden["dimensionless_extinction"][sample].get<double>(), 2e-11,
                     "C++ dimensionless extinction matches Python");
    }

    const double target = model.cumulativeHazard(coefficients, 1.0);
    context.near(model.inverseCumulativeHazard(coefficients, target), 1.0, 1e-12,
                 "cumulative hazard inversion recovers q");
    context.require(std::isinf(model.inverseCumulativeHazard(coefficients, 1000.0)),
                    "cumulative hazard inversion reports mass beyond q_max");
    context.near(model.transmittance(coefficients, 0.0), 1.0, 0.0,
                 "surface-birth transmittance starts at one");

    bool rejectedType = false;
    try {
        FirstPassageMlp invalid({"matern52_mlp", config.modulePath, config.bundlePath, "cpu"});
    } catch (const std::invalid_argument&) {
        rejectedType = true;
    }
    context.require(rejectedType, "kernel-specific MLP type names are rejected");
}

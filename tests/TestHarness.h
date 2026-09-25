#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

struct TestContext {
    int checks = 0;
    int failures = 0;

    void require(bool condition, const std::string& message) {
        ++checks;
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void near(double actual, double expected, double tolerance, const std::string& message) {
        require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
                message + " (actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected) + ")");
    }

    void relative(double actual, double expected, double tolerance, const std::string& message) {
        const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
        near(actual, expected, tolerance * scale, message);
    }
};

void testGaussianMath(TestContext& context);
void testGpssStatistics(TestContext& context);
void testMacrofacetBaseline(TestContext& context);
void testFlightKernels(TestContext& context);
void testSampling(TestContext& context);
void testRenderingThreads(TestContext& context);
void testRegularTracking(TestContext& context);
void testCutawayMean(TestContext& context);
void testShaderBallMean(TestContext& context);
void testMaterialRoughness(TestContext& context);

#if defined(MACROFACET_TEST_FIELDS)
void testNanoVdbField(TestContext& context);
#endif

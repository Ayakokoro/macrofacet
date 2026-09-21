#include "TestHarness.h"
#include <exception>
#include <iostream>

int main() {
    TestContext context;
    try { testGaussianMath(context); }
    catch (const std::exception& e) { context.require(false, std::string("Gaussian math threw: ") + e.what()); }
    try { testGpssStatistics(context); }
    catch (const std::exception& e) { context.require(false, std::string("GPSS statistics threw: ") + e.what()); }
    try { testMacrofacetBaseline(context); }
    catch (const std::exception& e) { context.require(false, std::string("baseline threw: ") + e.what()); }
    try { testFlightKernels(context); }
    catch (const std::exception& e) { context.require(false, std::string("flight kernels threw: ") + e.what()); }
    try { testSampling(context); }
    catch (const std::exception& e) { context.require(false, std::string("sampling threw: ") + e.what()); }
    try { testRenderingThreads(context); }
    catch (const std::exception& e) { context.require(false, std::string("render threading threw: ") + e.what()); }
    std::cout << "checks=" << context.checks << " failures=" << context.failures << '\n';
    return context.failures == 0 ? 0 : 1;
}

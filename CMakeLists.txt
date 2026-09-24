cmake_minimum_required(VERSION 3.24)
project(macrofacet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(macrofacet
    src/mathutility/Gaussian1D.cpp
    src/mathutility/GaussianMoments1D.cpp
    src/mathutility/BivariateGaussian.cpp
    src/mathutility/GaussianScreenIntegral.cpp
    src/gpss/MeanField.cpp
    src/gpss/ShaderBallMean.cpp
    src/gpss/SquaredExponentialKernel.cpp
    src/gpss/GPSSField.cpp
    src/gpss/ConditionedRay.cpp
    src/macrofacet/GaussianNdf.cpp
    src/macrofacet/GgxHeightfield.cpp
    src/macrofacet/ClassicCoefficients.cpp
    src/macrofacet/ConductorPhase.cpp
    src/transport/FlightState.cpp
    src/transport/FlightKernel.cpp
    src/transport/ClassicFlightKernel.cpp
    src/transport/Conditional29FlightKernel.cpp
    src/transport/MidpointFlightKernel.cpp
    src/transport/OpticalDepthSampler.cpp
    src/transport/ClassicNullTracking.cpp
    src/transport/CollisionGradientSampler.cpp
    src/integrator/MacrofacetPathTracer.cpp
    src/experiments/ExperimentConfig.cpp
    src/experiments/FlightCurveExperiment.cpp
    src/experiments/ConditionalGPReference.cpp
    src/experiments/RenderExperiment.cpp)

target_include_directories(macrofacet PUBLIC include)
target_link_libraries(macrofacet PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads)
if(MSVC)
    target_compile_options(macrofacet PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(macrofacet PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(macrofacet_experiments src/experiments/ExperimentMain.cpp)
target_link_libraries(macrofacet_experiments PRIVATE macrofacet)

add_executable(macrofacet_tests
    tests/test_main.cpp
    tests/test_gaussian_math.cpp
    tests/test_gpss_statistics.cpp
    tests/test_macrofacet_baseline.cpp
    tests/test_flight_kernels.cpp
    tests/test_sampling.cpp
    tests/test_cutaway_mean.cpp
    tests/test_shader_ball_mean.cpp
    tests/test_material_roughness.cpp
    tests/test_rendering_threads.cpp)
target_link_libraries(macrofacet_tests PRIVATE macrofacet)

enable_testing()
add_test(NAME macrofacet_tests COMMAND macrofacet_tests)

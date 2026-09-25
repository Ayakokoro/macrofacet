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
    src/fields/MeanFactory.cpp
    src/gpss/MeanField.cpp
    src/gpss/ShaderBallMean.cpp
    src/gpss/SquaredExponentialKernel.cpp
    src/gpss/GPSSField.cpp
    src/gpss/ConditionedRay.cpp
    src/macrofacet/GaussianNdf.cpp
    src/macrofacet/GgxHeightfield.cpp
    src/macrofacet/MaterialConfig.cpp
    src/macrofacet/ClassicCoefficients.cpp
    src/macrofacet/BeckmannVisibleSampler.cpp
    src/macrofacet/ConductorPhase.cpp
    src/transport/FlightState.cpp
    src/transport/FlightKernel.cpp
    src/transport/ClassicFlightKernel.cpp
    src/transport/Conditional29FlightKernel.cpp
    src/transport/OpticalDepthSampler.cpp
    src/transport/ClassicNullTracking.cpp
    src/transport/DensityMajorantGrid.cpp
    src/transport/NarrowBandMedium.cpp
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

# ---------------------------------------------------------------------------
# macrofacet_field: the NanoVDB-backed field library.
#
# Kept out of the `macrofacet` target on purpose -- the core library must not
# gain a nanovdb or libigl dependency. Consumers that link this library
# additionally get the "nanovdb" mean type registered.
# ---------------------------------------------------------------------------
option(MACROFACET_BUILD_FIELDS "Build the NanoVDB field generator" ON)

if(MACROFACET_BUILD_FIELDS)
    find_package(nanovdb CONFIG REQUIRED)
    find_package(libigl CONFIG REQUIRED)

    # The vcpkg libigl port does not install igl/raytri.c; see src/ext/igl_shim.
    add_library(igl_raytri_shim INTERFACE)
    target_include_directories(igl_raytri_shim SYSTEM INTERFACE src/ext/igl_shim)

    add_library(macrofacet_field
        src/fields/NanoVdbSampledField.cpp
        src/fields/NanoVdbMean.cpp
        src/fields/PrepareNanoVdbField.cpp
        src/fieldgen/PlyReader.cpp
        src/fieldgen/SyntheticMesh.cpp
        src/fieldgen/MeshDistanceField.cpp
        src/fieldgen/VdbBaker.cpp
        src/fieldgen/AnalyticVdbBaker.cpp
        src/fieldgen/NanoVdbIO.cpp)
    # `src` is PRIVATE: the PLY reader, the distance field and the baker are
    # library internals. Only include/ is public, matching the core library.
    target_include_directories(macrofacet_field PUBLIC include PRIVATE src)
    target_link_libraries(macrofacet_field
        PUBLIC macrofacet nanovdb::nanovdb igl::igl_core igl_raytri_shim)
    if(MSVC)
        # /utf-8 is mandatory: nanovdb headers contain non-ASCII bytes and the
        # default MSVC codepage here is 936, which produces a wall of C4819.
        # The core library is clean, so it keeps its own (narrower) flags.
        target_compile_options(macrofacet_field PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(macrofacet_field PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(macrofacet_fieldgen src/fieldgen/FieldGeneratorMain.cpp)
    target_include_directories(macrofacet_fieldgen PRIVATE src)
    target_link_libraries(macrofacet_fieldgen PRIVATE macrofacet_field)
    if(MSVC)
        target_compile_options(macrofacet_fieldgen PRIVATE /W4 /permissive- /EHsc /utf-8)
    endif()

    # Linking the field library is what makes a config's `"mean_type": "nanovdb"`
    # resolvable in a render run. The macro guards the explicit registration call
    # in ExperimentMain.cpp, so -DMACROFACET_BUILD_FIELDS=OFF still yields a
    # binary that builds and runs every procedural config.
    target_link_libraries(macrofacet_experiments PRIVATE macrofacet_field)
    target_compile_definitions(macrofacet_experiments PRIVATE MACROFACET_HAS_FIELDS=1)
endif()

set(MACROFACET_TEST_SOURCES
    tests/test_main.cpp
    tests/test_gaussian_math.cpp
    tests/test_gpss_statistics.cpp
    tests/test_macrofacet_baseline.cpp
    tests/test_flight_kernels.cpp
    tests/test_sampling.cpp
    tests/test_regular_tracking.cpp
    tests/test_cutaway_mean.cpp
    tests/test_shader_ball_mean.cpp
    tests/test_material_roughness.cpp
    tests/test_rendering_threads.cpp)
set(MACROFACET_TEST_LIBRARIES macrofacet)

# The field tests are only compiled when the field library exists; the rest of
# the suite must keep building with -DMACROFACET_BUILD_FIELDS=OFF.
if(MACROFACET_BUILD_FIELDS)
    list(APPEND MACROFACET_TEST_SOURCES tests/test_nanovdb_field.cpp)
    list(APPEND MACROFACET_TEST_LIBRARIES macrofacet_field)
endif()

add_executable(macrofacet_tests ${MACROFACET_TEST_SOURCES})
target_link_libraries(macrofacet_tests PRIVATE ${MACROFACET_TEST_LIBRARIES})
if(MACROFACET_BUILD_FIELDS)
    target_include_directories(macrofacet_tests PRIVATE src)
    target_compile_definitions(macrofacet_tests PRIVATE MACROFACET_TEST_FIELDS=1)
endif()

enable_testing()
add_test(NAME macrofacet_tests COMMAND macrofacet_tests)

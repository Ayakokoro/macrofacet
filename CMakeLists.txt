cmake_minimum_required(VERSION 3.24)
project(macrofacet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The Python wheel ships a complete LibTorch CMake package. Querying it here
# keeps the Python training runtime and the C++ inference ABI on exactly the
# same PyTorch build without requiring a second standalone LibTorch download.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import json; from pathlib import Path; import torch; print(json.dumps({'prefix': torch.utils.cmake_prefix_path, 'version': torch.__version__.split('+', 1)[0], 'runtime_directory': str(Path(torch.__file__).resolve().parent / 'lib')}))"
    RESULT_VARIABLE MACROFACET_TORCH_QUERY_RESULT
    OUTPUT_VARIABLE MACROFACET_TORCH_INFO
    ERROR_VARIABLE MACROFACET_TORCH_QUERY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT MACROFACET_TORCH_QUERY_RESULT EQUAL 0 OR NOT MACROFACET_TORCH_INFO)
    message(FATAL_ERROR
        "Unable to locate PyTorch/LibTorch using ${Python3_EXECUTABLE}.\n"
        "Install python/requirements-cuda.txt into that interpreter, or select "
        "another interpreter with -DPython3_EXECUTABLE=<python.exe>.\n"
        "${MACROFACET_TORCH_QUERY_ERROR}")
endif()
string(JSON MACROFACET_TORCH_PREFIX GET "${MACROFACET_TORCH_INFO}" prefix)
string(JSON MACROFACET_TORCH_VERSION GET "${MACROFACET_TORCH_INFO}" version)
string(JSON MACROFACET_TORCH_RUNTIME_DIRECTORY
    GET "${MACROFACET_TORCH_INFO}" runtime_directory)
file(TO_CMAKE_PATH "${MACROFACET_TORCH_PREFIX}" MACROFACET_TORCH_PREFIX)
file(TO_CMAKE_PATH "${MACROFACET_TORCH_RUNTIME_DIRECTORY}"
    MACROFACET_TORCH_RUNTIME_DIRECTORY)
if(MACROFACET_TORCH_VERSION VERSION_LESS "2.12" OR
   NOT MACROFACET_TORCH_VERSION VERSION_LESS "2.13")
    message(FATAL_ERROR
        "macrofacet requires PyTorch/LibTorch 2.12.x for C++17, but "
        "${Python3_EXECUTABLE} provides ${MACROFACET_TORCH_VERSION}")
endif()
set(Torch_DIR "${MACROFACET_TORCH_PREFIX}/Torch" CACHE PATH
    "LibTorch CMake package bundled with the selected Python interpreter" FORCE)
find_package(Torch 2.12 CONFIG REQUIRED)
message(STATUS
    "Using PyTorch/LibTorch ${MACROFACET_TORCH_VERSION} from ${Torch_DIR}")

add_library(macrofacet
    src/mathutility/Gaussian1D.cpp
    src/mathutility/GaussianMoments1D.cpp
    src/fields/MeanFactory.cpp
    src/gpss/MeanField.cpp
    src/gpss/ShaderBallMean.cpp
    src/gpss/CovarianceKernel.cpp
    src/gpss/GPSSField.cpp
    src/learned/FirstPassageMlp.cpp
    src/macrofacet/GaussianNdf.cpp
    src/macrofacet/GgxHeightfield.cpp
    src/macrofacet/MaterialConfig.cpp
    src/macrofacet/ClassicCoefficients.cpp
    src/macrofacet/BeckmannVisibleSampler.cpp
    src/macrofacet/ConductorPhase.cpp
    src/transport/FlightState.cpp
    src/transport/FlightKernel.cpp
    src/transport/ClassicFlightKernel.cpp
    src/transport/ClassicNullTracking.cpp
    src/transport/ConditionalFlightKernel.cpp
    src/transport/ConditionalMedium.cpp
    src/transport/ConditionalNullTracking.cpp
    src/transport/DensityMajorantGrid.cpp
    src/transport/NarrowBandMedium.cpp
    src/transport/CollisionGradientSampler.cpp
    src/integrator/MacrofacetPathTracer.cpp
    src/experiments/ExperimentConfig.cpp
    src/experiments/FirstPassageRice.cpp
    src/experiments/FirstPassageExperiment.cpp
    src/experiments/RenderExperiment.cpp)
target_sources(macrofacet PRIVATE src/experiments/TransmittanceCurves.cpp)

target_include_directories(macrofacet PUBLIC include)
target_link_libraries(macrofacet PUBLIC
    Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads ${TORCH_LIBRARIES})
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
    find_package(OpenVDB CONFIG REQUIRED)

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

    add_executable(macrofacet_nvdb_to_tungsten tools/nvdb_to_tungsten/main.cpp)
    target_link_libraries(macrofacet_nvdb_to_tungsten PRIVATE
        nanovdb::nanovdb OpenVDB::openvdb nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(macrofacet_nvdb_to_tungsten PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(macrofacet_nvdb_to_tungsten PRIVATE -Wall -Wextra -Wpedantic)
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
    tests/test_cutaway_mean.cpp
    tests/test_shader_ball_mean.cpp
    tests/test_material_roughness.cpp
    tests/test_first_passage_mlp.cpp
    tests/test_first_passage.cpp
    tests/test_transmittance_config.cpp
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

if(WIN32)
    # Windows resolves imported DLLs before main(), so a launcher must set the
    # search path before starting the executable. Reuse the Python wheel's DLLs
    # in place rather than copying them into every build configuration.
    set(MACROFACET_TORCH_EXECUTABLES macrofacet_experiments macrofacet_tests)
    if(MACROFACET_BUILD_FIELDS)
        list(APPEND MACROFACET_TORCH_EXECUTABLES macrofacet_fieldgen)
    endif()
    file(TO_NATIVE_PATH "${MACROFACET_TORCH_RUNTIME_DIRECTORY}"
        MACROFACET_TORCH_RUNTIME_DIRECTORY_NATIVE)
    foreach(executable IN LISTS MACROFACET_TORCH_EXECUTABLES)
        file(GENERATE
            OUTPUT "$<TARGET_FILE_DIR:${executable}>/${executable}.cmd"
            CONTENT "@echo off\nsetlocal\nset \"PATH=${MACROFACET_TORCH_RUNTIME_DIRECTORY_NATIVE};%PATH%\"\n\"$<TARGET_FILE:${executable}>\" %*\nexit /b %errorlevel%\n"
            NEWLINE_STYLE CRLF)
        set_target_properties(${executable} PROPERTIES
            VS_DEBUGGER_ENVIRONMENT
                "PATH=${MACROFACET_TORCH_RUNTIME_DIRECTORY_NATIVE};%PATH%")
    endforeach()
    message(STATUS
        "LibTorch runtime: generated .cmd launchers use ${MACROFACET_TORCH_RUNTIME_DIRECTORY}")
endif()

set(MACROFACET_MLP_TEST_DATA "${CMAKE_CURRENT_BINARY_DIR}/mlp_test_data")
file(MAKE_DIRECTORY "${MACROFACET_MLP_TEST_DATA}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/scripts/export_cpp_test_model.py"
            --output "${MACROFACET_MLP_TEST_DATA}"
    RESULT_VARIABLE MACROFACET_MLP_FIXTURE_RESULT
    ERROR_VARIABLE MACROFACET_MLP_FIXTURE_ERROR)
if(NOT MACROFACET_MLP_FIXTURE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create LibTorch test fixture: ${MACROFACET_MLP_FIXTURE_ERROR}")
endif()
file(TO_CMAKE_PATH "${MACROFACET_MLP_TEST_DATA}" MACROFACET_MLP_TEST_DATA_CMAKE)
target_compile_definitions(macrofacet_tests PRIVATE
    MACROFACET_MLP_TEST_DATA="${MACROFACET_MLP_TEST_DATA_CMAKE}")
if(MACROFACET_BUILD_FIELDS)
    target_include_directories(macrofacet_tests PRIVATE src)
    target_compile_definitions(macrofacet_tests PRIVATE MACROFACET_TEST_FIELDS=1)
endif()

enable_testing()
add_test(NAME macrofacet_tests COMMAND macrofacet_tests)
add_test(
    NAME macrofacet_data_analysis_tests
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s data_analysis/tests -v)
set_tests_properties(macrofacet_data_analysis_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
    set_tests_properties(macrofacet_tests PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${MACROFACET_TORCH_RUNTIME_DIRECTORY}")
endif()

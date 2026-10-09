cmake_minimum_required(VERSION 3.24)
project(macrofacet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_library(macrofacet
    src/mathutility/Gaussian1D.cpp
    src/mathutility/GaussianMoments1D.cpp
    src/fields/MeanFactory.cpp
    src/gpss/MeanField.cpp
    src/gpss/ShaderBallMean.cpp
    src/gpss/CovarianceKernel.cpp
    src/gpss/GPSSField.cpp
    src/mathutility/CubicPolynomial.cpp
    src/gpss/RayMeanProfile.cpp
    src/gpss/Matern32Reference.cpp
    src/learned/RenewalHazardModel.cpp
    src/learned/RenewalBatchSession.cpp
    src/transport/RenewalRayDistribution.cpp
    src/transport/RenewalMedium.cpp
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
    src/integrator/RenewalWavefront.cpp
    src/integrator/RenewalCpuExecutor.cpp
    src/experiments/ExperimentConfig.cpp
    src/experiments/FirstPassageRice.cpp
    src/experiments/FirstPassageExperiment.cpp
    src/experiments/RenewalModelExperiment.cpp
    src/experiments/RenderExperiment.cpp)
target_sources(macrofacet PRIVATE src/experiments/TransmittanceCurves.cpp)

target_include_directories(macrofacet PUBLIC include)
target_link_libraries(macrofacet PUBLIC
    Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads)

option(MACROFACET_ENABLE_TORCH "Enable LibTorch CPU/CUDA batched Renewal inference" ON)
if(MACROFACET_ENABLE_TORCH)
    if(NOT Torch_DIR)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib,torch; print(pathlib.Path(torch.__file__).parent/'share/cmake/Torch')"
            OUTPUT_VARIABLE MACROFACET_TORCH_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE MACROFACET_TORCH_DISCOVERY)
        if(MACROFACET_TORCH_DISCOVERY EQUAL 0)
            set(Torch_DIR "${MACROFACET_TORCH_DIR}" CACHE PATH "LibTorch CMake package")
        endif()
    endif()
    find_package(Torch CONFIG REQUIRED)
    # Isolate LibTorch's headers and C++20 requirement from the C++17 renderer.
    add_library(macrofacet_torch STATIC src/learned/RenewalTorchBackend.cpp)
    target_compile_features(macrofacet_torch PRIVATE cxx_std_20)
    target_link_libraries(macrofacet_torch PRIVATE ${TORCH_LIBRARIES})
    if(MSVC)
        target_compile_options(macrofacet_torch PRIVATE /EHsc /utf-8 /bigobj)
    endif()
    target_link_libraries(macrofacet PRIVATE macrofacet_torch)
    target_compile_definitions(macrofacet PRIVATE MACROFACET_HAS_TORCH=1)
endif()
if(MSVC)
    target_compile_options(macrofacet PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(macrofacet PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(macrofacet_experiments src/experiments/ExperimentMain.cpp)
target_link_libraries(macrofacet_experiments PRIVATE macrofacet)
add_executable(macrofacet_verify_renewal_batch EXCLUDE_FROM_ALL tools/verify_renewal_batch.cpp)
target_link_libraries(macrofacet_verify_renewal_batch PRIVATE macrofacet)

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
    # Optional profiling executable; no timers are added to the renderer.
    add_executable(macrofacet_renewal_benchmark EXCLUDE_FROM_ALL tools/benchmark_renewal.cpp)
    target_link_libraries(macrofacet_renewal_benchmark PRIVATE macrofacet_field)
    add_executable(macrofacet_inspect_renewal_ray EXCLUDE_FROM_ALL tools/inspect_renewal_ray.cpp)
    target_link_libraries(macrofacet_inspect_renewal_ray PRIVATE macrofacet_field)
    if(MSVC)
        target_compile_options(macrofacet_inspect_renewal_ray PRIVATE /EHsc /utf-8)
    endif()
    if(MACROFACET_ENABLE_TORCH)
        # Diagnostic copies override the corresponding static-library objects.
        # The production executable has no profiling scopes or CUDA barriers.
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(RENEWAL_PROFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/renewal_profile")
            add_custom_command(
                OUTPUT "${RENEWAL_PROFILE_DIR}/RenewalWavefrontProfile.cpp"
                       "${RENEWAL_PROFILE_DIR}/RenewalTorchProfile.cpp"
                       "${RENEWAL_PROFILE_DIR}/RenewalBatchSessionProfile.cpp"
                       "${RENEWAL_PROFILE_DIR}/RayMeanProfileProfile.cpp"
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/profiling/generate_renewal_profile.py"
                        "${CMAKE_CURRENT_SOURCE_DIR}" "${RENEWAL_PROFILE_DIR}"
                DEPENDS tools/profiling/generate_renewal_profile.py
                        src/integrator/RenewalWavefront.cpp src/learned/RenewalTorchBackend.cpp
                        src/learned/RenewalBatchSession.cpp src/gpss/RayMeanProfile.cpp
                VERBATIM)
            add_executable(macrofacet_profile_renewal EXCLUDE_FROM_ALL
                tools/profiling/profile_renewal_main.cpp
                "${RENEWAL_PROFILE_DIR}/RenewalWavefrontProfile.cpp"
                "${RENEWAL_PROFILE_DIR}/RenewalTorchProfile.cpp"
                "${RENEWAL_PROFILE_DIR}/RenewalBatchSessionProfile.cpp"
                "${RENEWAL_PROFILE_DIR}/RayMeanProfileProfile.cpp")
            target_include_directories(macrofacet_profile_renewal PRIVATE tools/profiling src/learned)
            target_compile_features(macrofacet_profile_renewal PRIVATE cxx_std_20)
            target_compile_definitions(macrofacet_profile_renewal PRIVATE MACROFACET_HAS_TORCH=1)
            target_link_libraries(macrofacet_profile_renewal PRIVATE macrofacet_field ${TORCH_LIBRARIES} CUDA::cudart)
            if(MSVC)
                target_compile_options(macrofacet_profile_renewal PRIVATE /EHsc /utf-8 /bigobj)
            endif()
        endif()
    endif()
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
    tests/test_renewal_reference.cpp
    tests/test_renewal_transport.cpp
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
    # Keep direct .exe and .cmd launches usable without a Python process.
    set(MACROFACET_EXECUTABLES macrofacet_experiments macrofacet_tests)
    if(MACROFACET_BUILD_FIELDS)
        list(APPEND MACROFACET_EXECUTABLES macrofacet_fieldgen)
    endif()
    foreach(executable IN LISTS MACROFACET_EXECUTABLES)
        file(GENERATE OUTPUT "$<TARGET_FILE_DIR:${executable}>/${executable}.cmd"
            CONTENT "@echo off\n\"%~dp0${executable}.exe\" %*\nexit /b %errorlevel%\n"
            NEWLINE_STYLE CRLF)
    endforeach()
endif()

if(WIN32 AND MACROFACET_ENABLE_TORCH)
    file(GLOB MACROFACET_TORCH_DLLS CONFIGURE_DEPENDS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    set(MACROFACET_TORCH_RUNTIME_STAMP "${CMAKE_CURRENT_BINARY_DIR}/torch-runtime-$<CONFIG>.stamp")
    add_custom_command(OUTPUT "${MACROFACET_TORCH_RUNTIME_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:macrofacet_experiments>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${MACROFACET_TORCH_DLLS}
            "$<TARGET_FILE_DIR:macrofacet_experiments>"
        COMMAND "${CMAKE_COMMAND}" -E touch "${MACROFACET_TORCH_RUNTIME_STAMP}"
        DEPENDS ${MACROFACET_TORCH_DLLS}
        COMMENT "Deploying LibTorch runtime DLLs" VERBATIM)
    add_custom_target(macrofacet_torch_runtime DEPENDS "${MACROFACET_TORCH_RUNTIME_STAMP}")
    foreach(executable macrofacet_experiments macrofacet_tests macrofacet_fieldgen macrofacet_renewal_benchmark macrofacet_verify_renewal_batch macrofacet_profile_renewal macrofacet_inspect_renewal_ray)
        if(TARGET ${executable})
            add_dependencies(${executable} macrofacet_torch_runtime)
        endif()
    endforeach()
endif()

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

# Python neural tests remain optional, independently of the C++ LibTorch backend.
option(MACROFACET_TEST_NEURAL "Run Renewal+ Python tests (requires torch and numpy)" OFF)
if(MACROFACET_TEST_NEURAL)
    add_test(NAME macrofacet_renewal_tests
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s python/tests -v)
    set_tests_properties(macrofacet_renewal_tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        ENVIRONMENT "MACROFACET_RENEWAL_EXE=$<TARGET_FILE:macrofacet_experiments>")
endif()

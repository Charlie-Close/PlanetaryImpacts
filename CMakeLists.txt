cmake_minimum_required(VERSION 3.24)

project(sph_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPH_VULKAN_BUILD_SHADERS "Compile GLSL shaders to SPIR-V during the build" ON)
option(SPH_VULKAN_BUILD_VIEWER "Build the non-headless GLFW viewer" ON)

find_package(Vulkan REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
if (SPH_VULKAN_BUILD_VIEWER)
    find_package(glfw3 3.3 REQUIRED)
endif()

add_executable(sph_vulkan
    src/Aneos.cpp
    src/Hdf5.cpp
    src/Main.cpp
    src/Octree.cpp
    src/Simulator.cpp
    src/VulkanContext.cpp
    src/VulkanSimulation.cpp
    third_party/lodepng.cpp
)

target_include_directories(sph_vulkan PRIVATE
    include
    third_party
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(sph_vulkan PRIVATE
    Vulkan::Vulkan
    ${HDF5_C_LIBRARIES}
)

if (SPH_VULKAN_BUILD_VIEWER)
    target_compile_definitions(sph_vulkan PRIVATE SPH_HAS_VIEWER=1 GLFW_INCLUDE_VULKAN)
    target_sources(sph_vulkan PRIVATE src/Viewer.cpp)
    target_link_libraries(sph_vulkan PRIVATE glfw)
else()
    target_compile_definitions(sph_vulkan PRIVATE SPH_HAS_VIEWER=0)
endif()

target_compile_definitions(sph_vulkan PRIVATE
    SPH_PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
)

if (MSVC)
    target_compile_options(sph_vulkan PRIVATE /W4 /permissive-)
else()
    target_compile_options(sph_vulkan PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()

find_program(GLSLC_EXECUTABLE glslc)
if (SPH_VULKAN_BUILD_SHADERS AND GLSLC_EXECUTABLE)
    file(READ "${CMAKE_CURRENT_LIST_DIR}/include/sph/Parameters.hpp" SPH_PARAMETERS_HPP)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/include/sph/Parameters.hpp"
    )

    function(sph_extract_param NAME OUT_VAR)
        string(REGEX MATCH "inline constexpr (float|int) ${NAME} = ([^\n;]+);" _MATCH "${SPH_PARAMETERS_HPP}")
        if (NOT _MATCH)
            message(FATAL_ERROR "Could not extract sph::params::${NAME} for shader generation")
        endif()
        set(_VALUE "${CMAKE_MATCH_2}")
        string(REPLACE "params::" "" _VALUE "${_VALUE}")
        string(REPLACE "f" "" _VALUE "${_VALUE}")
        string(STRIP "${_VALUE}" _VALUE)
        set(${OUT_VAR} "${_VALUE}" PARENT_SCOPE)
    endfunction()

    sph_extract_param(viscosityAlpha SPH_VISCOSITY_ALPHA)
    sph_extract_param(viscosityBeta SPH_VISCOSITY_BETA)
    sph_extract_param(alphaMin SPH_ALPHA_MIN)
    sph_extract_param(alphaMax SPH_ALPHA_MAX)
    sph_extract_param(densityEta SPH_DENSITY_ETA)
    sph_extract_param(maxSmoothingLength SPH_MAX_SMOOTHING_LENGTH)
    sph_extract_param(maxDensityIterations SPH_MAX_DENSITY_NR_ITERATIONS)
    sph_extract_param(neighboursEstimate SPH_N_NEIGHBOURS_ESTIM)
    sph_extract_param(resolutionEta SPH_RESOLUTION_ETA)
    sph_extract_param(particleInRangeThreshold SPH_PARTICLE_IN_RANGE_THRESHOLD)
    sph_extract_param(cfl SPH_CFL)
    sph_extract_param(gdt SPH_GDT)
    sph_extract_param(minDt SPH_MIN_DT)
    sph_extract_param(maxDt SPH_MAX_DT)
    sph_extract_param(gamma SPH_GAMMA)
    sph_extract_param(kernelConstant SPH_KERNEL_CONSTANT)
    sph_extract_param(gravitySmoothingLength SPH_GRAVITY_SMOOTHING_LENGTH)
    sph_extract_param(plumberEquivalent SPH_PLUMBER_EQUIVALENT)
    sph_extract_param(gravityEta SPH_GRAVITY_ETA)
    sph_extract_param(multipoleExpansionPower SPH_MULTIPOLE_EXPANSION_POWER)
    if (SPH_MULTIPOLE_EXPANSION_POWER LESS 1 OR SPH_MULTIPOLE_EXPANSION_POWER GREATER 4)
        message(FATAL_ERROR "Vulkan multipole expansion currently supports powers 1 through 4")
    endif()
    math(EXPR SPH_N_EXPANSION_TERMS
        "(${SPH_MULTIPOLE_EXPANSION_POWER} + 1) * (${SPH_MULTIPOLE_EXPANSION_POWER} + 2) * (${SPH_MULTIPOLE_EXPANSION_POWER} + 3) / 6"
    )
    math(EXPR SPH_N_MULTIPOLE_POWERS "${SPH_MULTIPOLE_EXPANSION_POWER} + 1")
    sph_extract_param(gravityMaxRecursion SPH_GRAVITY_MAX_RECURSION)
    sph_extract_param(maxFloat SPH_MAX_FLOAT)
    sph_extract_param(cellWidth SPH_CELL_WIDTH)
    sph_extract_param(cellPower SPH_CELL_POWER)
    sph_extract_param(sortingBlockSize SPH_SORTING_BLOCK_SIZE)
    sph_extract_param(sortingMaskLength SPH_SORTING_MASK_LENGTH)
    math(EXPR SPH_SORTING_BUCKET_NUMBER "1 << ${SPH_SORTING_MASK_LENGTH}")
    math(EXPR SPH_SORTING_BIT_MASK "${SPH_SORTING_BUCKET_NUMBER} - 1")

    string(REGEX MATCH "inline constexpr Vec3 lightDirection = \\{([^,]+), ([^,]+), ([^,]+)," _LIGHT_MATCH "${SPH_PARAMETERS_HPP}")
    if (NOT _LIGHT_MATCH)
        message(FATAL_ERROR "Could not extract sph::params::lightDirection for shader generation")
    endif()
    set(SPH_LIGHT_X "${CMAKE_MATCH_1}")
    set(SPH_LIGHT_Y "${CMAKE_MATCH_2}")
    set(SPH_LIGHT_Z "${CMAKE_MATCH_3}")
    string(REPLACE "f" "" SPH_LIGHT_X "${SPH_LIGHT_X}")
    string(REPLACE "f" "" SPH_LIGHT_Y "${SPH_LIGHT_Y}")
    string(REPLACE "f" "" SPH_LIGHT_Z "${SPH_LIGHT_Z}")
    string(STRIP "${SPH_LIGHT_X}" SPH_LIGHT_X)
    string(STRIP "${SPH_LIGHT_Y}" SPH_LIGHT_Y)
    string(STRIP "${SPH_LIGHT_Z}" SPH_LIGHT_Z)

    set(SPH_GENERATED_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_shaders")
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/shaders/sph_params.glsl.in"
        "${SPH_GENERATED_SHADER_DIR}/sph_params.glsl"
        @ONLY
    )

    file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/shaders/*.comp"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/*.vert"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/*.frag"
    )
    set(SPIRV_OUTPUTS)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME "${SHADER}" NAME)
        set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
            COMMAND "${GLSLC_EXECUTABLE}" -I "${SPH_GENERATED_SHADER_DIR}" "${SHADER}" -o "${SPIRV}"
            DEPENDS "${SHADER}" "${SPH_GENERATED_SHADER_DIR}/sph_params.glsl"
            VERBATIM
        )
        list(APPEND SPIRV_OUTPUTS "${SPIRV}")
    endforeach()
    add_custom_target(sph_vulkan_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
    add_dependencies(sph_vulkan sph_vulkan_shaders)
elseif(SPH_VULKAN_BUILD_SHADERS)
    message(WARNING "glslc was not found; C++ target will build, but shaders will not be precompiled.")
endif()

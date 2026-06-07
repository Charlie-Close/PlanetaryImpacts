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
            COMMAND "${GLSLC_EXECUTABLE}" "${SHADER}" -o "${SPIRV}"
            DEPENDS "${SHADER}"
            VERBATIM
        )
        list(APPEND SPIRV_OUTPUTS "${SPIRV}")
    endforeach()
    add_custom_target(sph_vulkan_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
    add_dependencies(sph_vulkan sph_vulkan_shaders)
elseif(SPH_VULKAN_BUILD_SHADERS)
    message(WARNING "glslc was not found; C++ target will build, but shaders will not be precompiled.")
endif()

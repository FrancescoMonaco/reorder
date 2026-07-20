cmake_minimum_required(VERSION 3.25)
Include(FetchContent)

project(REORDER CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add warnings
add_compile_options(-Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function)
# Always optimize as much as we can
# Instead of relying on -march=native (which breaks when building with nix) we
# define the flags we need here
add_compile_options(-O3 -g -march=native) #-mavx2 -mfma)

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(icecream-cpp)

include_directories("include")
include_directories("include/external/icecream-cpp")

add_executable(reorder test/reorder_test.cpp)
target_link_libraries(reorder PRIVATE icecream-cpp::icecream-cpp)


###############################################################################
# Python Bindings
# 
# This cmake code is adapted from
# - https://nanobind.readthedocs.io/en/latest/building.html
# - https://nanobind.readthedocs.io/en/latest/packaging.html
# Explicitly find the Conda Python interpreter first if active
if(DEFINED ENV{CONDA_PREFIX})
  set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()

find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_reorder_impl NOMINSIZE NB_STATIC python/wrapper.cpp)
target_link_libraries(_reorder_impl PRIVATE)

# --- NEW: AUTOMATICALLY COPY TO THE PYTHON SCRIPT FOLDER ---
# This forces the compiled .so/.pyd to be created directly in your source tree (e.g. inside CLUB)
# so python reorder.py can immediately import it without any copying.
set_target_properties(_reorder_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/CLUB"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/CLUB" # For Windows DLLs/.pyd
)

install(TARGETS _reorder_impl LIBRARY DESTINATION reorder)

###############################################################################
# Clang-Format Integration
###############################################################################
find_program(CLANG_FORMAT_BIN NAMES clang-format)

if(CLANG_FORMAT_BIN)
    # Gather all C++ source and header files (excluding external libraries)
    file(GLOB_RECURSE SRC_FILES 
        "include/club/*.hpp"
        "python/*.cpp"
        "test/*.cpp"
    )

    # Create a custom target that runs clang-format on the files
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_BIN} -i ${SRC_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on source files..."
    )

    # Make build targets depend on the format target 
    # so it runs BEFORE they compile
    add_dependencies(reorder format)
    add_dependencies(_reorder_impl format)
else()
    message(WARNING "clang-format not found. Auto-formatting is disabled.")
endif()
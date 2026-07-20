cmake_minimum_required(VERSION 3.12)
project(MtxPermSparseBase VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)
# Find SparseBase
set(SPARSEBASE_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sparsebase_install")
list(APPEND CMAKE_PREFIX_PATH "${SPARSEBASE_INSTALL_DIR}")

find_package(sparsebase REQUIRED PATHS "${SPARSEBASE_INSTALL_DIR}")

# Common compiler flags
if(NOT MSVC)
    add_compile_options(-O3 -fopenmp)
    add_link_options(-fopenmp)
else()
    add_compile_options(/O2 /openmp)
endif()

# Automatically compile all .cpp files in src/
file(GLOB REORDERING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")


# Timers
add_subdirectory("ccutils")

# Detect which reorderers are available in SparseBase by trying to compile a test
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${sparsebase_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES sparsebase::sparsebase)

# Test for AMD
check_cxx_source_compiles("
#include <sparsebase/reorder/amd_reorder.h>
int main() {
    sparsebase::reorder::AMDReorderParams params;
    return 0;
}" HAVE_AMD_REORDER)

# Test for METIS
check_cxx_source_compiles("
#include <sparsebase/reorder/metis_reorder.h>
int main() {
    sparsebase::reorder::MetisReorderParams params;
    return 0;
}" HAVE_METIS_REORDER)

# Test for METIS partition
check_cxx_source_compiles("
#include <sparsebase/partition/metis_partition.h>
int main() {
    sparsebase::partition::MetisPartitionParams params;
    return 0;
}" HAVE_METIS_PARTITION)

# Test for Rabbit
check_cxx_source_compiles("
#include <sparsebase/reorder/rabbit_reorder.h>
int main() {
    sparsebase::reorder::RabbitReorderParams params;
    return 0;
}" HAVE_RABBIT_REORDER)

# Test for PaToH
check_cxx_source_compiles("
#include <sparsebase/partition/patoh_partition.h>
int main() {
    sparsebase::partition::PatohPartitionParams params;
    return 0;
}" HAVE_PATOH_PARTITION)

# Test for SlashBurn
check_cxx_source_compiles("
#include <sparsebase/reorder/slashburn_reorder.h>
int main() {
    sparsebase::reorder::SlashburnReorderParams params;
    return 0;
}" HAVE_SLASHBURN_REORDER)

set(TOOL_TARGETS "")
foreach(SOURCE_FILE ${REORDERING_SOURCES})
    # Get filename without extension
    get_filename_component(TOOL_NAME ${SOURCE_FILE} NAME_WE)
    
    # Check if this tool requires unavailable features
    set(SKIP_TOOL FALSE)
    if(TOOL_NAME STREQUAL "amd_perm" AND NOT HAVE_AMD_REORDER)
        message(STATUS "  Skipping: ${TOOL_NAME} (AMD reorder not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    elseif(TOOL_NAME STREQUAL "metis_perm" AND NOT HAVE_METIS_REORDER)
        message(STATUS "  Skipping: ${TOOL_NAME} (METIS reorder not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    elseif(TOOL_NAME STREQUAL "metis_part_perm" AND NOT HAVE_METIS_PARTITION)
        message(STATUS "  Skipping: ${TOOL_NAME} (METIS partition not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    elseif(TOOL_NAME STREQUAL "rabbit_perm" AND NOT HAVE_RABBIT_REORDER)
        message(STATUS "  Skipping: ${TOOL_NAME} (Rabbit reorder not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    elseif(TOOL_NAME STREQUAL "patoh_perm" AND NOT HAVE_PATOH_PARTITION)
        message(STATUS "  Skipping: ${TOOL_NAME} (PaToH partition not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    elseif(TOOL_NAME STREQUAL "slashburn_perm" AND NOT HAVE_SLASHBURN_REORDER)
        message(STATUS "  Skipping: ${TOOL_NAME} (SlashBurn reorder not available in SparseBase)")
        set(SKIP_TOOL TRUE)
    endif()
    
    if(NOT SKIP_TOOL)
        # Create executable
        add_executable(${TOOL_NAME} ${SOURCE_FILE})
        target_link_libraries(${TOOL_NAME} PRIVATE ccutils::ccutils sparsebase::sparsebase)
        
        # Add to list for installation
        list(APPEND TOOL_TARGETS ${TOOL_NAME})
        
        message(STATUS "  Building: ${TOOL_NAME}")
    endif()
endforeach()

# Installation
if(TOOL_TARGETS)
    install(TARGETS ${TOOL_TARGETS} RUNTIME DESTINATION bin)
endif()

# Print build information
message(STATUS "SparseBase Tools Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  SparseBase Location: ${SPARSEBASE_INSTALL_DIR}")

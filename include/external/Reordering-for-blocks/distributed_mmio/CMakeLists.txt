cmake_minimum_required(VERSION 3.10)
project(distributed_mmio LANGUAGES C CXX)

option(DMMIO_ENABLE_MPI "Enable MPI for distributed reading" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


### CCUTILS dependency ###
set(CCUTILS_ENABLE_MPI  ON  CACHE BOOL "")
include(FetchContent)
FetchContent_Declare(
    ccutils
    GIT_REPOSITORY https://github.com/ThomasPasquali/ccutils.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(ccutils)

### Distributed MMIO library ###
add_library(dmmio STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio/mmio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c/wrapper.cpp)
target_include_directories(dmmio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(dmmio PRIVATE ccutils::ccutils)

### MTX to BMTX converter ###
option(DMMIO_TOOLS "Build mtx_to_bmtx tool" OFF)
if(DMMIO_TOOLS)
    add_executable(mtx_to_bmtx ${CMAKE_CURRENT_SOURCE_DIR}/src/tools/mtx_to_bmtx.cpp)
    target_link_libraries(mtx_to_bmtx PRIVATE dmmio ccutils::ccutils)
endif()

# ---------------- MPI (Optional) ----------------
if(DMMIO_ENABLE_MPI)
    message("DMMIO: Enabling MPI module")
    find_package(MPI REQUIRED COMPONENTS C CXX)  # Provides MPI::MPI_C and MPI::MPI_CXX
    target_sources(dmmio PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dmmio/dmmio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dmmio/dio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dmmio/partitioning.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dmmio/dutils.cpp
    )
    target_link_libraries(dmmio PRIVATE MPI::MPI_C MPI::MPI_CXX ccutils::ccutils_mpi)
    if(DMMIO_TOOLS)
        target_link_libraries(mtx_to_bmtx PRIVATE MPI::MPI_C MPI::MPI_CXX ccutils::ccutils_mpi)
    endif()
endif()
cmake_minimum_required(VERSION 3.20)
project(kahypar_perm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# mt-kahypar is installed into a local prefix by install.sh.
# Allow overriding the location, e.g. -DMTKAHYPAR_DIR=/path/to/install/lib/cmake/MtKaHyPar
set(MTKAHYPAR_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mt-kahypar-install"
    CACHE PATH "mt-kahypar install prefix")
list(APPEND CMAKE_PREFIX_PATH "${MTKAHYPAR_INSTALL_DIR}")

# mt-kahypar exports the package name `MtKaHyPar` and the imported target
# `MtKaHyPar::mtkahypar`. That target's INTERFACE link libraries reference
# TBB::tbb / TBB::tbbmalloc, and MtKaHyPar's config does NOT call
# find_dependency(TBB) itself, so we must locate TBB first.
find_package(TBB REQUIRED)
find_package(MtKaHyPar REQUIRED CONFIG)

add_executable(kahypar_perm kahypar_perm.cpp)
target_link_libraries(kahypar_perm PRIVATE MtKaHyPar::mtkahypar TBB::tbb)
